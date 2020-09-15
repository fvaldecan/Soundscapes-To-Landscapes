function opts = birdOpts()
opts = delimitedTextImportOptions("NumVariables", 20);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["filename", "recording", "site", "device", "year"...
    , "month", "day", "hour", "min", "species", "birdcode", "commonname"...
    , "songtype", "x1", "x2", "y1", "y2", "score", "vote", "type"];
opts.VariableTypes = ["string", "string", "string", "string", "double"...
    , "double", "double", "double", "double", "string", "string", "string"...
    , "string", "double", "double", "double", "double", "double", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["filename", "recording"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["filename", "recording", "site", "device", "species", "birdcode"...
    , "commonname", "songtype", "vote", "type"], "EmptyFieldRule", "auto");
end

