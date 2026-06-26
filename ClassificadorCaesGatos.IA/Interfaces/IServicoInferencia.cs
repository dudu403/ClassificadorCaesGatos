using ClassificadorCaesGatos.IA.Dtos;

namespace ClassificadorCaesGatos.IA.Interfaces;

public interface IServicoInferencia
{
    Task<ResultadoPredicao> PredizerAsync(
        Stream imagem,
        string caminhoModelo,
        CancellationToken cancellationToken);
}