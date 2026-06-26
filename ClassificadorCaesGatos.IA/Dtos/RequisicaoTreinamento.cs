namespace ClassificadorCaesGatos.IA.Dtos;

public sealed class RequisicaoTreinamento
{
    public int Epocas { get; set; } = 20;

    public int BatchSize { get; set; } = 16;

    public double LearningRate { get; set; } = 0.001;
}