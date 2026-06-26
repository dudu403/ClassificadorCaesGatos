namespace ClassificadorCaesGatos.IA.Avaliacao;

public static class Metricas
{
    public static double CalcularAcuracia(int acertos, int total)
    {
        if (total == 0)
            return 0;

        return (double)acertos / total;
    }

    public static double CalcularPrecisao(int verdadeiroPositivo, int falsoPositivo)
    {
        int denominador = verdadeiroPositivo + falsoPositivo;

        if (denominador == 0)
            return 0;

        return (double)verdadeiroPositivo / denominador;
    }

    public static double CalcularRecall(int verdadeiroPositivo, int falsoNegativo)
    {
        int denominador = verdadeiroPositivo + falsoNegativo;

        if (denominador == 0)
            return 0;

        return (double)verdadeiroPositivo / denominador;
    }

    public static double CalcularF1Score(double precisao, double recall)
    {
        double denominador = precisao + recall;

        if (denominador == 0)
            return 0;

        return 2 * precisao * recall / denominador;
    }
}