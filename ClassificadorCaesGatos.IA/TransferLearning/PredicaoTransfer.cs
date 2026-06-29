using Microsoft.ML.Data;

namespace ClassificadorCaesGatos.IA.TransferLearning;

public sealed class PredicaoTransfer
{
    [ColumnName("PredictedLabel")]
    public string RotuloPrevisto { get; set; } = string.Empty;

    public float[] Score { get; set; } = [];
}