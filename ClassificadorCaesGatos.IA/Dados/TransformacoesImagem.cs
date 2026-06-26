using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TorchSharp;
using static TorchSharp.torch;

namespace ClassificadorCaesGatos.IA.Dados;

public sealed class TransformacoesImagem
{
    public Tensor CarregarImagemComoTensor(
        string caminhoImagem,
        bool aplicarDataAugmentation,
        int tamanhoImagem = 224)
    {
        using Image<Rgb24> imagem = Image.Load<Rgb24>(caminhoImagem);

        imagem.Mutate(contexto =>
        {
            contexto.Resize(tamanhoImagem, tamanhoImagem);
        });

        float[] pixels = new float[3 * tamanhoImagem * tamanhoImagem];

        for (int y = 0; y < tamanhoImagem; y++)
        {
            for (int x = 0; x < tamanhoImagem; x++)
            {
                Rgb24 pixel = imagem[x, y];

                int indice = y * tamanhoImagem + x;

                pixels[indice] = pixel.R / 255f;
                pixels[tamanhoImagem * tamanhoImagem + indice] = pixel.G / 255f;
                pixels[2 * tamanhoImagem * tamanhoImagem + indice] = pixel.B / 255f;
            }
        }

        Tensor imagemTensor = torch.tensor(pixels, dtype: ScalarType.Float32);

        return imagemTensor.reshape(3, tamanhoImagem, tamanhoImagem);
    }
}