namespace ClassificadorCaesGatos.IA.Dados;

public sealed class DatasetImagens
{
    private static readonly string[] ExtensoesPermitidas =
    [
        ".jpg",
        ".jpeg",
        ".png"
    ];

    public List<AmostraImagem> CarregarDataset(string caminhoBase)
    {
        if (!Directory.Exists(caminhoBase))
            throw new DirectoryNotFoundException($"Pasta não encontrada: {caminhoBase}");

        List<AmostraImagem> amostras = [];

        CarregarClasse(amostras, caminhoBase, "gato", 0);
        CarregarClasse(amostras, caminhoBase, "cachorro", 1);

        if (amostras.Count == 0)
            throw new InvalidOperationException($"Nenhuma imagem encontrada em: {caminhoBase}");

        return amostras;
    }

    private static void CarregarClasse(
        List<AmostraImagem> amostras,
        string caminhoBase,
        string nomeClasse,
        int classe)
    {
        string caminhoClasse = Path.Combine(caminhoBase, nomeClasse);

        if (!Directory.Exists(caminhoClasse))
            throw new DirectoryNotFoundException($"Pasta da classe não encontrada: {caminhoClasse}");

        List<string> arquivos = Directory
            .EnumerateFiles(caminhoClasse, "*.*", SearchOption.AllDirectories)
            .Where(arquivo => ExtensoesPermitidas.Contains(Path.GetExtension(arquivo).ToLowerInvariant()))
            .ToList();

        foreach (string arquivo in arquivos)
        {
            amostras.Add(new AmostraImagem
            {
                Caminho = arquivo,
                Classe = classe,
                NomeClasse = nomeClasse
            });
        }
    }
}