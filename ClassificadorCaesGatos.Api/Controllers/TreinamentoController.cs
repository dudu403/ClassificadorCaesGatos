using ClassificadorCaesGatos.IA.Dtos;
using ClassificadorCaesGatos.IA.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace ClassificadorCaesGatos.Api.Controllers;

[ApiController]
[Route("api/treinamento")]
public sealed class TreinamentoController : ControllerBase
{
    private readonly IServicoTreinamento _servicoTreinamento;
    private readonly IWebHostEnvironment _environment;

    public TreinamentoController(
        IServicoTreinamento servicoTreinamento,
        IWebHostEnvironment environment)
    {
        _servicoTreinamento = servicoTreinamento;
        _environment = environment;
    }

    [HttpPost("iniciar")]
    public async Task<ActionResult<RespostaTreinamento>> IniciarAsync(
        [FromBody] RequisicaoTreinamento requisicao,
        CancellationToken cancellationToken)
    {
        string caminhoDataset = Path.Combine(_environment.ContentRootPath, "dataset");
        string caminhoModelo = Path.Combine(_environment.ContentRootPath, "modelos", "modelo.pt");

        ResultadoTreinamento resultado = await _servicoTreinamento.TreinarAsync(
            caminhoDataset,
            caminhoModelo,
            requisicao.Epocas,
            requisicao.BatchSize,
            requisicao.LearningRate,
            cancellationToken);

        return Ok(new RespostaTreinamento
        {
            Sucesso = true,
            Mensagem = "Treinamento concluído com sucesso.",
            Resultado = resultado
        });
    }
}