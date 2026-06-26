using TorchSharp;
using static TorchSharp.torch;

namespace ClassificadorCaesGatos.IA.Treinamento;

public sealed class ConfiguracaoTreinamento
{
    public int Epocas { get; set; } = 20;

    public int BatchSize { get; set; } = 32;

    public double LearningRate { get; set; } = 0.001;

    public int QuantidadeClasses { get; set; } = 2;

    public int TamanhoImagem { get; set; } = 224;

    public bool EmbaralharDataset { get; set; } = true;

    public Device Device { get; set; } = torch.CPU;
}