namespace ClassificadorCaesGatos.IA.Dtos;

public sealed class RespostaPredicao
{
    public string Classe { get; set; } = string.Empty;

    public double Confianca { get; set; }
}