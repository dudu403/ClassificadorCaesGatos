using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ClassificadorCaesGatos.IA.RedeNeural;

public sealed class RedeConvolucional : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> _features;
    private readonly Module<Tensor, Tensor> _classificador;

    public RedeConvolucional() : base(nameof(RedeConvolucional))
    {
        _features = Sequential(
            Conv2d(3, 32, kernel_size: 3, stride: 1, padding: 1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size: 2, stride: 2),

            Conv2d(32, 64, kernel_size: 3, stride: 1, padding: 1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size: 2, stride: 2),

            Conv2d(64, 128, kernel_size: 3, stride: 1, padding: 1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size: 2, stride: 2),

            Conv2d(128, 256, kernel_size: 3, stride: 1, padding: 1),
            BatchNorm2d(256),
            ReLU(),
            AdaptiveAvgPool2d(new long[] { 1, 1 })
        );

        _classificador = Sequential(
            Flatten(),
            Dropout(0.5),
            Linear(256, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 2)
        );

        RegisterComponents();
    }

    public override Tensor forward(Tensor entrada)
    {
        Tensor saida = _features.forward(entrada);
        saida = _classificador.forward(saida);

        return saida;
    }
}