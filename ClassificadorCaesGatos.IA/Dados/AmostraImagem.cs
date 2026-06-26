using TorchSharp;
using static TorchSharp.torch;

namespace ClassificadorCaesGatos.IA.Dados;

public sealed class AmostraImagem
{
    public string Caminho { get; set; } = string.Empty;

    public int Classe { get; set; }

    public string NomeClasse { get; set; } = string.Empty;

    public Tensor? Tensor { get; set; }
}