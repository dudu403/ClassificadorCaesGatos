using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace ClassificadorCaesGatos.IA.TransferLearning;

public sealed class TreinadorTransferLearning
{
    private readonly MLContext _mlContext = new(seed: 1);

    public Task<object> TreinarAsync(
        string caminhoDataset,
        string caminhoModelo,
        int epocas,
        int batchSize,
        float learningRate,
        CancellationToken cancellationToken)
    {
        return Task.Run<object>(() =>
        {
            string caminhoTreino = ObterPasta(caminhoDataset, "treino", "train");
            string caminhoValidacao = ObterPasta(caminhoDataset, "validacao", "validation");
            string caminhoTeste = ObterPasta(caminhoDataset, "teste", "test");

            List<ImagemEntradaTransfer> treino = CarregarImagens(caminhoTreino);
            List<ImagemEntradaTransfer> validacao = CarregarImagens(caminhoValidacao);
            List<ImagemEntradaTransfer> teste = CarregarImagens(caminhoTeste);

            IDataView dadosTreino = _mlContext.Data.LoadFromEnumerable(treino);
            IDataView dadosValidacao = _mlContext.Data.LoadFromEnumerable(validacao);

            string workspacePath = Path.Combine(
                Path.GetDirectoryName(caminhoModelo)!,
                "workspace-transfer-learning");

            Directory.CreateDirectory(workspacePath);

            var preProcessamento = _mlContext.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "LabelAsKey",
                    inputColumnName: nameof(ImagemEntradaTransfer.Rotulo))
                .Append(_mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: string.Empty,
                    inputColumnName: nameof(ImagemEntradaTransfer.CaminhoImagem)));

            ITransformer preProcessamentoTreinado = preProcessamento.Fit(dadosTreino);

            IDataView dadosValidacaoPreparados = preProcessamentoTreinado.Transform(dadosValidacao);

            var pipeline = preProcessamento
                .Append(_mlContext.MulticlassClassification.Trainers.ImageClassification(
                    new ImageClassificationTrainer.Options
                    {
                        FeatureColumnName = "Image",
                        LabelColumnName = "LabelAsKey",
                        ValidationSet = dadosValidacaoPreparados,
                        Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                        Epoch = epocas,
                        BatchSize = batchSize,
                        LearningRate = learningRate,
                        WorkspacePath = workspacePath,
                        MetricsCallback = metricas =>
                        {
                            if (metricas.Train is not null)
                            {
                                Console.WriteLine(
                                    $"Transfer Learning -> Época: {metricas.Train.Epoch} | " +
                                    $"Acurácia treino: {metricas.Train.Accuracy:P2}");
                            }
                        }
                    }))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            ITransformer modelo = pipeline.Fit(dadosTreino);

            Directory.CreateDirectory(Path.GetDirectoryName(caminhoModelo)!);

            _mlContext.Model.Save(
                modelo,
                dadosTreino.Schema,
                caminhoModelo);

            double acuraciaTeste = CalcularAcuraciaTeste(modelo, teste);

            return new
            {
                Success = true,
                Mensagem = "Transfer Learning concluído com sucesso.",
                Arquitetura = "ResNetV2-50",
                Treino = treino.Count,
                Validacao = validacao.Count,
                Teste = teste.Count,
                Epocas = epocas,
                BatchSize = batchSize,
                LearningRate = learningRate,
                AcuraciaTeste = acuraciaTeste,
                CaminhoModelo = caminhoModelo
            };
        }, cancellationToken);
    }

    public PredicaoTransfer Predizer(string caminhoImagem, string caminhoModelo)
    {
        if (!File.Exists(caminhoModelo))
            throw new FileNotFoundException("Modelo de transfer learning não encontrado.", caminhoModelo);

        ITransformer modelo = _mlContext.Model.Load(caminhoModelo, out _);

        PredictionEngine<ImagemEntradaTransfer, PredicaoTransfer> engine =
            _mlContext.Model.CreatePredictionEngine<ImagemEntradaTransfer, PredicaoTransfer>(modelo);

        return engine.Predict(new ImagemEntradaTransfer
        {
            CaminhoImagem = caminhoImagem
        });
    }

    private double CalcularAcuraciaTeste(
        ITransformer modelo,
        List<ImagemEntradaTransfer> teste)
    {
        PredictionEngine<ImagemEntradaTransfer, PredicaoTransfer> engine =
            _mlContext.Model.CreatePredictionEngine<ImagemEntradaTransfer, PredicaoTransfer>(modelo);

        int acertos = 0;

        foreach (ImagemEntradaTransfer imagem in teste)
        {
            PredicaoTransfer predicao = engine.Predict(imagem);

            if (predicao.RotuloPrevisto.Equals(imagem.Rotulo, StringComparison.OrdinalIgnoreCase))
                acertos++;
        }

        return teste.Count == 0 ? 0 : (double)acertos / teste.Count;
    }

    private static List<ImagemEntradaTransfer> CarregarImagens(string caminhoBase)
    {
        if (!Directory.Exists(caminhoBase))
            throw new DirectoryNotFoundException($"Pasta não encontrada: {caminhoBase}");

        List<ImagemEntradaTransfer> imagens = [];

        foreach (string pastaClasse in Directory.GetDirectories(caminhoBase))
        {
            string nomeClasse = Path.GetFileName(pastaClasse);
            string rotulo = NormalizarRotulo(nomeClasse);

            string[] arquivos = Directory
                .GetFiles(pastaClasse, "*.*", SearchOption.AllDirectories)
                .Where(arquivo =>
                    arquivo.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                    arquivo.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                    arquivo.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                .ToArray();

            foreach (string arquivo in arquivos)
            {
                imagens.Add(new ImagemEntradaTransfer
                {
                    CaminhoImagem = arquivo,
                    Rotulo = rotulo
                });
            }
        }

        if (imagens.Count == 0)
            throw new InvalidOperationException($"Nenhuma imagem encontrada em: {caminhoBase}");

        return imagens;
    }

    private static string ObterPasta(
        string caminhoDataset,
        string nomePortugues,
        string nomeIngles)
    {
        string caminhoPortugues = Path.Combine(caminhoDataset, nomePortugues);

        if (Directory.Exists(caminhoPortugues))
            return caminhoPortugues;

        string caminhoIngles = Path.Combine(caminhoDataset, nomeIngles);

        if (Directory.Exists(caminhoIngles))
            return caminhoIngles;

        throw new DirectoryNotFoundException(
            $"Não encontrei a pasta '{nomePortugues}' nem '{nomeIngles}' dentro de {caminhoDataset}");
    }

    private static string NormalizarRotulo(string nomeClasse)
    {
        return nomeClasse.ToLowerInvariant() switch
        {
            "cat" => "gato",
            "gato" => "gato",
            "dog" => "cachorro",
            "cachorro" => "cachorro",
            _ => nomeClasse
        };
    }
}