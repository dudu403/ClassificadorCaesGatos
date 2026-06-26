namespace ClassificadorCaesGatos.IA.Avaliacao;

public sealed class MatrizConfusao
{
    public int[,] Valores { get; } = new int[2, 2];

    public void Adicionar(int classeReal, int classePredita)
    {
        Valores[classeReal, classePredita]++;
    }

    public int VerdadeiroGato => Valores[0, 0];

    public int GatoComoCachorro => Valores[0, 1];

    public int CachorroComoGato => Valores[1, 0];

    public int VerdadeiroCachorro => Valores[1, 1];
}