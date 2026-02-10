using CSV
using DataFrames
using Plots

filename = "data/weather.csv"
df = CSV.read(filename, DataFrame)
idx = [2,3,11,12]
data = Float64.(Array(df)[:,idx])

figure = plot(data[:,4])