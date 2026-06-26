using ClassificadorCaesGatos.IA.Dtos;

namespace ClassificadorCaesGatos.IA.Interfaces;

public interface IServicoTreinamento
{
    Task<ResultadoTreinamento> TreinarAsync(
        string caminhoDataset,
        string caminhoModelo,
        int epocas,
        int batchSize,
        double learningRate,
        CancellationToken cancellationToken);
}