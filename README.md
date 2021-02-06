# Environment setup

## Start jupyter

<!-- #region -->
```bash
VERSION=2.0.5
JUPYTER_SOURCES=$HOME/sources/
 
sudo docker run --net host \
    -v $JUPYTER_SOURCES:/home/runner/notebooks \
    -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/credentials/$GOOGLE_APPLICATION_SERVICEFILE \
    -e GIT_EMAIL="$GIT_EMAIL" -e GIT_USERNAME="$GIT_USERNAME" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/credentials/$GOOGLE_APPLICATION_SERVICEFILE \
    -e PYTHONPATH="/home/runner/notebooks/fiducias/src/:/home/runner/notebooks/fiducias/financial_ml/" \
    -t thepandorasys/jupyter-tools:$VERSION
```
<!-- #endregion -->

```python

```

# Repository setup

El codigo usa ell archivo de propiedades para poder determinar las rutas de todos los componentes.

<!-- #region -->
```bash
git clone https://github.com/SebastianCerquera/openseries.git

cat > .env <<EOF
REPO=$(pwd)
EOF
```
<!-- #endregion -->

```python

```

## Run financial ml unit tests


<!-- #region -->
```bash
cd financial_ml
python -m unittest test.test_core
```
<!-- #endregion -->

```python

```

# Usage

<!-- #region -->
### EDA

```bash
extract_features(){
    local NOTEBOOK_NAME=$1
    local NOTEBOOK_SERIE_NAME=$2

    local NOTEBOOK_FIDUCIA="FONDO DE INVERSION COLECTIVA ABIERTO RENTA ALTA CONVICCION"
    local NOTEBOOK_FILENAME="valores-bancolombia-fic_2020-11-29.csv"
    local NOTEBOOK_EXECUTION_DATE="2020-11-29"
         
    local OUTPUT_BASENAME=generated/$NOTEBOOK_EXECUTION_DATE/$(echo $NOTEBOOK_FIDUCIA |  perl -ne 's/ /_/g && print $_')/
    local OUTPUT_FILENAME="$NOTEBOOK_NAME"__$(echo $NOTEBOOK_SERIE_NAME | perl -ne 's/ /_/g; s/\.//g; s/á/a/g; s/é/e/g; s/í/i/g; s/ó/o/g; s/ú/u/g; print $_')
     
    mkdir -p notebooks/$OUTPUT_BASENAME
                                                                                       
    python utils/run_notebook.py \
    --input_notebook notebooks/$NOTEBOOK_NAME.ipynb \
    --output_notebook notebooks/$OUTPUT_BASENAME/$OUTPUT_FILENAME.ipynb \
    --notebook_fiducia "$NOTEBOOK_FIDUCIA" \
    --notebook_filename "$NOTEBOOK_FILENAME" \
    --notebook_execution_date "$NOTEBOOK_EXECUTION_DATE" \
    --notebook_serie_name "$NOTEBOOK_SERIE_NAME"
}


for j in "Núm. Invers." "Núm. unidades" "Valor unidad para las operaciones del día t" "Valor fondo al cierre del día t"; do
    for i in "unit_value___EDA" "1st_variation___EDA" "2nd_variation___EDA"; do
       extract_features "$i" "$j"
    done
done
```
<!-- #endregion -->

```python

```
