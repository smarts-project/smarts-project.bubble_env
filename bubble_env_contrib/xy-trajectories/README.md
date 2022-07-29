This should contain i80 text format files from NGSIM. If you wish to use scl build to generate the shf files for the ngsim related text scenarios from https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj.

From this you will want to do the following:
- Download `I80-Emeryville-CA.zip`
- Unpack `i-80-vehicle-trajectory-data.zip` from `I80-Emeryville-CA.zip`
- Unpack  `i-80-vehicle-trajectory-data.zip`
- Place `vehicle-trajectory-data` in `i80/`

Some suggested commands are:
```bash
curl -o I80-Emeryville-CA.zip https://data.transportation.gov/api/views/8ect-6jqj/files/10ed9f47-ab08-4df20b46a-29f1683ceffa?download=true&filename=I-80-Emeryville-CA.zip
unzip I80-Emeryville-CA.zip -d tmp_dir
unzip tmp_dir/i-80-vehicle-trajectory-data.zip -d tmp_dir
rsync -a --prune-empty-dirs --include '*/' --include '*.txt' --exclude '*' tmp_dir/ i80/
rm tmp_dir
```