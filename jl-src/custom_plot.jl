using Plots
using LaTeXStrings
using Measures

# Global publication theme
default(
    fontfamily = "Computer Modern",
    linewidth = 1.2,
    markersize = 4,
    legendfontsize = 9,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 14,
    framestyle = :box,
    grid = false,
    minorgrid = false,
    tickdirection = :out,
    foreground_color_border = :black,
    foreground_color_axis = :black,
    foreground_color_text = :black,
    background_color = :white,
    size = (720, 480),
    dpi = 500,
    margin = 7.5mm,
    fillalpha = 10.0,
    fillcolor = :blue,
)

# Consistent color cycle (colorblind-safe, print-friendly)
const PUB_COLORS = [
    RGB(0.0, 0.2, 0.6),   # deep blue
    RGB(0.8, 0.2, 0.2),   # red
    RGB(0.2, 0.6, 0.2),   # green
    RGB(0.6, 0.4, 0.0),   # ochre
    RGB(0.4, 0.2, 0.6)    # purple
]
palette(PUB_COLORS)

# Convenience wrapper for axis labels (LaTeX by default)
xlabel!(s) = xlabel!(L"$s$")
ylabel!(s) = ylabel!(L"$s$")