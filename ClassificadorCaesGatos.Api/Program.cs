using ClassificadorCaesGatos.IA.Inferencia;
using ClassificadorCaesGatos.IA.Interfaces;
using ClassificadorCaesGatos.IA.TransferLearning;
using ClassificadorCaesGatos.IA.Treinamento;

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddScoped<IServicoTreinamento, TreinadorModelo>();
builder.Services.AddScoped<IServicoInferencia, ServicoInferencia>();
builder.Services.AddScoped<TreinadorTransferLearning>();

WebApplication app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.MapControllers();

app.Run();