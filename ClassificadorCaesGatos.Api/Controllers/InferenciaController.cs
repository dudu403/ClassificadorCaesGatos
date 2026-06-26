using ClassificadorCaesGatos.IA.Dtos;
using ClassificadorCaesGatos.IA.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace ClassificadorCaesGatos.Api.Controllers;

[ApiController]
[Route("api/inferencia")]
public sealed class InferenciaController : ControllerBase
{
    private readonly IServicoInferencia _servicoInferencia;
    private readonly IWebHostEnvironment _environment;

    public InferenciaController(
        IServicoInferencia servicoInferencia,
        IWebHostEnvironment environment)
    {
        _servicoInferencia = servicoInferencia;
        _environment = environment;
    }

    [HttpPost("predizer")]
    [Consumes("multipart/form-data")]
    public async Task<ActionResult<RespostaPredicao>> PredizerAsync(
        IFormFile imagem,
        CancellationToken cancellationToken)
    {
        if (imagem is null || imagem.Length == 0)
            return BadRequest("Imagem inválida.");

        string caminhoModelo = Path.Combine(
            _environment.ContentRootPath,
            "modelos",
            "modelo.pt");

        if (!System.IO.File.Exists(caminhoModelo))
            return BadRequest("Modelo ainda não foi treinado.");

        await using Stream stream = imagem.OpenReadStream();

        ResultadoPredicao resultado = await _servicoInferencia.PredizerAsync(
            stream,
            caminhoModelo,
            cancellationToken);

        return Ok(new RespostaPredicao
        {
            Classe = resultado.Classe,
            Confianca = resultado.Confianca
        });
    }
}