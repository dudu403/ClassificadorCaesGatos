using ClassificadorCaesGatos.IA.Configuracoes;
using ClassificadorCaesGatos.IA.Dados;
using ClassificadorCaesGatos.IA.Dtos;
using ClassificadorCaesGatos.IA.Interfaces;
using ClassificadorCaesGatos.IA.RedeNeural;
using TorchSharp;
using static TorchSharp.torch;

namespace ClassificadorCaesGatos.IA.Inferencia;

public sealed class ServicoInferencia : IServicoInferencia
{
    private readonly TransformacoesImagem _transformacoesImagem;
    private readonly GerenciadorDispositivo _gerenciadorDispositivo;

    public ServicoInferencia()
    {
        _transformacoesImagem = new TransformacoesImagem();

        ConfiguracaoCuda configuracaoCuda = new()
        {
            UsarGpu = true,
            PermitirCpu = true
        };

        _gerenciadorDispositivo = new GerenciadorDispositivo(configuracaoCuda);
    }

    public async Task<ResultadoPredicao> PredizerAsync(
        Stream imagem,
        string caminhoModelo,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(caminhoModelo))
            throw new FileNotFoundException("Modelo treinado não encontrado.", caminhoModelo);

        string arquivoTemporario = Path.Combine(
            Path.GetTempPath(),
            $"{Guid.NewGuid():N}.jpg");

        await using (FileStream fileStream = File.Create(arquivoTemporario))
        {
            await imagem.CopyToAsync(fileStream, cancellationToken);
        }

        try
        {
            Device device = _gerenciadorDispositivo.ObterDispositivo();

            RedeConvolucional modelo = new();
            modelo.load(caminhoModelo);
            modelo.to(device);
            modelo.eval();

            using Tensor imagemTensor = _transformacoesImagem
                .CarregarImagemComoTensor(
                    arquivoTemporario,
                    aplicarDataAugmentation: false)
                .unsqueeze(0)
                .to(device);

            using IDisposable semGradiente = no_grad();

            using Tensor saida = modelo.forward(imagemTensor);
            using Tensor probabilidades = saida.softmax(1);
            using Tensor classePreditaTensor = probabilidades.argmax(1);

            long classePredita = classePreditaTensor
                .cpu()
                .data<long>()
                .First();

            float[] probabilidadesArray = probabilidades
                .cpu()
                .data<float>()
                .ToArray();

            double confianca = probabilidadesArray[classePredita] * 100;

            return new ResultadoPredicao
            {
                Classe = classePredita == 0 ? "Gato" : "Cachorro",
                Confianca = Math.Round(confianca, 2)
            };
        }
        finally
        {
            if (File.Exists(arquivoTemporario))
                File.Delete(arquivoTemporario);
        }
    }
}