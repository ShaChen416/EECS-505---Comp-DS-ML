function imshow(X::Matrix; kw_args...)
    return heatmap(
        X;
        color=:grays,
        yflip=true,
        aspect_ratio=:equal,
        axis=false,
        ticks=false,
        colorbar=false,
        kw_args...
    )
end