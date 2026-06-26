using System;
using TorchSharp;

namespace ClassificadorCaesGatos.IA.Configuracoes;

public sealed class ConfiguracaoCuda
{
    public bool UsarGpu { get; set; }

    public string Dispositivo { get; set; } = string.Empty;

    public bool PermitirCpu { get; set; } = true;
}