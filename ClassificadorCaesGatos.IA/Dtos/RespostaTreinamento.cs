namespace ClassificadorCaesGatos.IA.Dtos;

public sealed class RespostaTreinamento
{
    public bool Sucesso { get; set; }

    public string Mensagem { get; set; } = string.Empty;

    public ResultadoTreinamento? Resultado { get; set; }
}