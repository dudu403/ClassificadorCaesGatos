namespace ClassificadorCaesGatos.IA.Dtos;

public sealed class ResultadoPredicao
{
    public string Classe { get; set; } = string.Empty;

    public double Confianca { get; set; }
}