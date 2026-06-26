using TorchSharp;
using static TorchSharp.torch;

namespace ClassificadorCaesGatos.IA.Configuracoes;

public sealed class GerenciadorDispositivo
{
    private readonly ConfiguracaoCuda _configuracao;

    public GerenciadorDispositivo(ConfiguracaoCuda configuracao)
    {
        _configuracao = configuracao;
    }

    public Device ObterDispositivo()
    {
        if (_configuracao.UsarGpu && cuda.is_available())
        {
            _configuracao.Dispositivo = "CUDA";

            return CUDA;
        }

        if (_configuracao.PermitirCpu)
        {
            _configuracao.Dispositivo = "CPU";

            return CPU;
        }

        throw new InvalidOperationException("Nenhum dispositivo disponível para treinamento.");
    }

    public bool GpuDisponivel()
    {
        return cuda.is_available();
    }

    public string ObterNomeDispositivo()
    {
        if (cuda.is_available())
            return "CUDA";

        return "CPU";
    }
}