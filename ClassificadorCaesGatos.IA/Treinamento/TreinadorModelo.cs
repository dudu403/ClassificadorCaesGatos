using ClassificadorCaesGatos.IA.Avaliacao;
using ClassificadorCaesGatos.IA.Configuracoes;
using ClassificadorCaesGatos.IA.Dados;
using ClassificadorCaesGatos.IA.Dtos;
using ClassificadorCaesGatos.IA.Interfaces;
using ClassificadorCaesGatos.IA.RedeNeural;
using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClassificadorCaesGatos.IA.Treinamento;

public sealed class TreinadorModelo : IServicoTreinamento
{
    private readonly DatasetImagens _datasetImagens;
    private readonly TransformacoesImagem _transformacoesImagem;
    private readonly GerenciadorDispositivo _gerenciadorDispositivo;
    private bool _debugImpresso;

    public TreinadorModelo()
    {
        _datasetImagens = new DatasetImagens();
        _transformacoesImagem = new TransformacoesImagem();

        ConfiguracaoCuda configuracaoCuda = new()
        {
            UsarGpu = true,
            PermitirCpu = true
        };

        _gerenciadorDispositivo = new GerenciadorDispositivo(configuracaoCuda);
    }

    public Task<ResultadoTreinamento> TreinarAsync(
        string caminhoDataset,
        string caminhoModelo,
        int epocas,
        int batchSize,
        double learningRate,
        CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            _debugImpresso = false;

            Device device = _gerenciadorDispositivo.ObterDispositivo();

            string caminhoTreino = Path.Combine(caminhoDataset, "treino");
            string caminhoValidacao = Path.Combine(caminhoDataset, "validacao");
            string caminhoTeste = Path.Combine(caminhoDataset, "teste");

            List<AmostraImagem> treino = _datasetImagens.CarregarDataset(caminhoTreino);
            List<AmostraImagem> validacao = _datasetImagens.CarregarDataset(caminhoValidacao);
            List<AmostraImagem> teste = _datasetImagens.CarregarDataset(caminhoTeste);

            Console.WriteLine("Iniciando treinamento...");
            Console.WriteLine($"Dispositivo utilizado: {device.type}");
            Console.WriteLine($"Treino: {treino.Count} imagens");
            Console.WriteLine($"Validação: {validacao.Count} imagens");
            Console.WriteLine($"Teste: {teste.Count} imagens");
            Console.WriteLine($"Épocas: {epocas}");
            Console.WriteLine($"BatchSize: {batchSize}");
            Console.WriteLine($"LearningRate: {learningRate}");

            RedeConvolucional modelo = new();
            modelo.to(device);

            Loss<Tensor, Tensor, Tensor> criterio = CrossEntropyLoss();
            optim.Optimizer otimizador = optim.Adam(modelo.parameters(), learningRate);

            Stopwatch cronometro = Stopwatch.StartNew();

            double lossTreino = 0;
            double acuraciaTreino = 0;
            double lossValidacao = 0;
            double acuraciaValidacao = 0;

            for (int epoca = 1; epoca <= epocas; epoca++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                Console.WriteLine();
                Console.WriteLine($"========== ÉPOCA {epoca}/{epocas} ==========");

                treino = treino.OrderBy(_ => Random.Shared.Next()).ToList();

                (lossTreino, acuraciaTreino) = TreinarEpoca(
                    modelo,
                    treino,
                    batchSize,
                    criterio,
                    otimizador,
                    device,
                    cancellationToken);

                Console.WriteLine($"Treino -> Loss: {lossTreino:F4} | Acurácia: {acuraciaTreino:P2}");

                (lossValidacao, acuraciaValidacao, _) = Avaliar(
                    modelo,
                    validacao,
                    batchSize,
                    criterio,
                    device,
                    cancellationToken);

                Console.WriteLine($"Validação -> Loss: {lossValidacao:F4} | Acurácia: {acuraciaValidacao:P2}");
            }

            Console.WriteLine();
            Console.WriteLine("Executando teste final...");

            (double lossTeste, double acuraciaTeste, MatrizConfusao matrizConfusao) = Avaliar(
                modelo,
                teste,
                batchSize,
                criterio,
                device,
                cancellationToken);

            cronometro.Stop();

            Console.WriteLine($"Teste -> Loss: {lossTeste:F4} | Acurácia: {acuraciaTeste:P2}");
            Console.WriteLine("Salvando modelo...");

            Directory.CreateDirectory(Path.GetDirectoryName(caminhoModelo)!);
            modelo.save(caminhoModelo);

            Console.WriteLine($"Modelo salvo em: {caminhoModelo}");
            Console.WriteLine("Treinamento concluído.");

            return new ResultadoTreinamento
            {
                LossTreino = lossTreino,
                AcuraciaTreino = acuraciaTreino,
                LossValidacao = lossValidacao,
                AcuraciaValidacao = acuraciaValidacao,
                LossTeste = lossTeste,
                AcuraciaTeste = acuraciaTeste,
                TempoTreinamento = cronometro.Elapsed,
                MatrizConfusao = ConverterMatriz(matrizConfusao.Valores),
                CaminhoModelo = caminhoModelo
            };
        }, cancellationToken);
    }

    private (double lossMedia, double acuracia) TreinarEpoca(
        RedeConvolucional modelo,
        List<AmostraImagem> amostras,
        int batchSize,
        Loss<Tensor, Tensor, Tensor> criterio,
        optim.Optimizer otimizador,
        Device device,
        CancellationToken cancellationToken)
    {
        modelo.train();

        double somaLoss = 0;
        int total = 0;
        int acertos = 0;
        int quantidadeBatches = 0;

        foreach (List<AmostraImagem> batch in CriarBatches(amostras, batchSize))
        {
            cancellationToken.ThrowIfCancellationRequested();

            using Tensor imagensCpu = CriarTensorImagens(batch, aplicarDataAugmentation: true);
            using Tensor labelsCpu = CriarTensorLabels(batch);

            if (!_debugImpresso)
            {
                ImprimirDebugBatch(batch, imagensCpu, labelsCpu);
                _debugImpresso = true;
            }

            using Tensor imagens = imagensCpu.to(device);
            using Tensor labels = labelsCpu.to(device);

            otimizador.zero_grad();

            using Tensor saidas = modelo.forward(imagens);
            using Tensor loss = criterio.call(saidas, labels);

            loss.backward();
            otimizador.step();

            somaLoss += loss.ToSingle();
            quantidadeBatches++;

            using Tensor predicoes = saidas.argmax(1);

            long[] predicoesArray = predicoes.cpu().data<long>().ToArray();
            long[] labelsArray = labels.cpu().data<long>().ToArray();

            for (int i = 0; i < labelsArray.Length; i++)
            {
                if (predicoesArray[i] == labelsArray[i])
                    acertos++;

                total++;
            }
        }

        return (
            somaLoss / Math.Max(1, quantidadeBatches),
            Metricas.CalcularAcuracia(acertos, total)
        );
    }

    private (double lossMedia, double acuracia, MatrizConfusao matrizConfusao) Avaliar(
        RedeConvolucional modelo,
        List<AmostraImagem> amostras,
        int batchSize,
        Loss<Tensor, Tensor, Tensor> criterio,
        Device device,
        CancellationToken cancellationToken)
    {
        modelo.eval();

        double somaLoss = 0;
        int total = 0;
        int acertos = 0;
        int quantidadeBatches = 0;

        MatrizConfusao matrizConfusao = new();

        using IDisposable noGrad = no_grad();

        foreach (List<AmostraImagem> batch in CriarBatches(amostras, batchSize))
        {
            cancellationToken.ThrowIfCancellationRequested();

            using Tensor imagens = CriarTensorImagens(batch, aplicarDataAugmentation: false).to(device);
            using Tensor labels = CriarTensorLabels(batch).to(device);

            using Tensor saidas = modelo.forward(imagens);
            using Tensor loss = criterio.call(saidas, labels);
            using Tensor predicoes = saidas.argmax(1);

            somaLoss += loss.ToSingle();
            quantidadeBatches++;

            long[] predicoesArray = predicoes.cpu().data<long>().ToArray();
            long[] labelsArray = labels.cpu().data<long>().ToArray();

            for (int i = 0; i < labelsArray.Length; i++)
            {
                int classeReal = (int)labelsArray[i];
                int classePredita = (int)predicoesArray[i];

                if (classeReal == classePredita)
                    acertos++;

                matrizConfusao.Adicionar(classeReal, classePredita);
                total++;
            }
        }

        return (
            somaLoss / Math.Max(1, quantidadeBatches),
            Metricas.CalcularAcuracia(acertos, total),
            matrizConfusao
        );
    }

    private Tensor CriarTensorImagens(
        List<AmostraImagem> batch,
        bool aplicarDataAugmentation)
    {
        Tensor[] imagens = batch
            .Select(amostra => _transformacoesImagem.CarregarImagemComoTensor(
                amostra.Caminho,
                aplicarDataAugmentation))
            .ToArray();

        Tensor tensorBatch = stack(imagens);

        foreach (Tensor imagem in imagens)
        {
            imagem.Dispose();
        }

        return tensorBatch;
    }

    private static Tensor CriarTensorLabels(List<AmostraImagem> batch)
    {
        long[] labels = batch
            .Select(amostra => (long)amostra.Classe)
            .ToArray();

        return tensor(labels, dtype: ScalarType.Int64);
    }

    private static IEnumerable<List<AmostraImagem>> CriarBatches(
        List<AmostraImagem> amostras,
        int batchSize)
    {
        for (int i = 0; i < amostras.Count; i += batchSize)
        {
            yield return amostras
                .Skip(i)
                .Take(batchSize)
                .ToList();
        }
    }

    private static void ImprimirDebugBatch(
        List<AmostraImagem> batch,
        Tensor imagens,
        Tensor labels)
    {
        Console.WriteLine();
        Console.WriteLine("========== DEBUG DO PRIMEIRO BATCH ==========");
        Console.WriteLine($"Shape imagens CPU: {string.Join(",", imagens.shape)}");
        Console.WriteLine($"Shape labels CPU: {string.Join(",", labels.shape)}");

        long[] labelsArray = labels.cpu().data<long>().ToArray();

        Console.WriteLine($"Labels: {string.Join(",", labelsArray)}");

        Console.WriteLine("Amostras:");
        foreach (AmostraImagem amostra in batch.Take(5))
        {
            Console.WriteLine($"{amostra.NomeClasse} -> {amostra.Classe} -> {Path.GetFileName(amostra.Caminho)}");
        }

        Console.WriteLine("============================================");
        Console.WriteLine();
    }

    private static int[][] ConverterMatriz(int[,] matriz)
    {
        return
        [
            [matriz[0, 0], matriz[0, 1]],
            [matriz[1, 0], matriz[1, 1]]
        ];
    }
}