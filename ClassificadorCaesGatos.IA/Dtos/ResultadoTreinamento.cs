namespace ClassificadorCaesGatos.IA.Dtos;

public sealed class ResultadoTreinamento
{
    public double LossTreino { get; set; }

    public double AcuraciaTreino { get; set; }

    public double LossValidacao { get; set; }

    public double AcuraciaValidacao { get; set; }

    public double LossTeste { get; set; }

    public double AcuraciaTeste { get; set; }

    public TimeSpan TempoTreinamento { get; set; }

    public int[][] MatrizConfusao { get; set; } = [];

    public string CaminhoModelo { get; set; } = string.Empty;
}