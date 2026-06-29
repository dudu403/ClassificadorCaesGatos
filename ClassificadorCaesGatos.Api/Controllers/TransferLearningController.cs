using ClassificadorCaesGatos.IA.TransferLearning;
using Microsoft.AspNetCore.Mvc;

namespace ClassificadorCaesGatos.Api.Controllers;

[ApiController]
[Route("api/transfer-learning")]
public sealed class TransferLearningController : ControllerBase
{
    private readonly TreinadorTransferLearning _treinadorTransferLearning;
    private readonly IWebHostEnvironment _environment;

    public TransferLearningController(
        TreinadorTransferLearning treinadorTransferLearning,
        IWebHostEnvironment environment)
    {
        _treinadorTransferLearning = treinadorTransferLearning;
        _environment = environment;
    }

    [HttpPost("treinar")]
    public async Task<ActionResult<object>> TreinarAsync(
        int epocas = 10,
        int batchSize = 16,
        float learningRate = 0.01f,
        CancellationToken cancellationToken = default)
    {
        string caminhoDataset = Path.Combine(
            _environment.ContentRootPath,
            "dataset");

        string caminhoModelo = Path.Combine(
            _environment.ContentRootPath,
            "modelos",
            "modelo-transfer-learning.zip");

        object resultado = await _treinadorTransferLearning.TreinarAsync(
            caminhoDataset,
            caminhoModelo,
            epocas,
            batchSize,
            learningRate,
            cancellationToken);

        return Ok(resultado);
    }

    [HttpPost("predizer")]
    [Consumes("multipart/form-data")]
    public async Task<ActionResult<object>> PredizerAsync(
        IFormFile imagem,
        CancellationToken cancellationToken)
    {
        if (imagem is null || imagem.Length == 0)
            return BadRequest("Imagem inválida.");

        string caminhoModelo = Path.Combine(
            _environment.ContentRootPath,
            "modelos",
            "modelo-transfer-learning.zip");

        if (!System.IO.File.Exists(caminhoModelo))
            return BadRequest("Modelo de transfer learning ainda não foi treinado.");

        string arquivoTemporario = Path.Combine(
            Path.GetTempPath(),
            $"{Guid.NewGuid():N}{Path.GetExtension(imagem.FileName)}");

        await using (FileStream fileStream = System.IO.File.Create(arquivoTemporario))
        {
            await imagem.CopyToAsync(fileStream, cancellationToken);
        }

        try
        {
            PredicaoTransfer predicao = _treinadorTransferLearning.Predizer(
                arquivoTemporario,
                caminhoModelo);

            float confianca = predicao.Score.Length == 0
                ? 0
                : predicao.Score.Max() * 100;

            return Ok(new
            {
                Classe = predicao.RotuloPrevisto,
                Confianca = Math.Round(confianca, 2),
                Scores = predicao.Score
            });
        }
        finally
        {
            if (System.IO.File.Exists(arquivoTemporario))
                System.IO.File.Delete(arquivoTemporario);
        }
    }
}