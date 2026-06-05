# Thesis PCG Terrain Generation

No CLI interface yet.

To generate terrain, run one of the main scripts:

```bash
python3 -m pcg.diamond_square
```

or

```bash
python3 -m pcg.perlin_noise
```

Inside the scripts themselves, comment or uncomment the function you want to run.

## Diamond-Square

```python
ds_gen_full_basic()
```

Diamond-square full terrain.

```python
ds_gen_full_hashed()
```

Diamond-square full terrain with hash instead of PRNG.

```python
ds_gen_single()
```

Diamond-square single coordinate query.

```python
ds_gen_rect()
```

Diamond-square single coordinate looped over rectangle.

## Perlin Noise

```python
pn_gen_full_basic()
```

Perlin noise full grid terrain.

```python
pn_gen_full_octaves()
```

Perlin noise full grid terrain with octaves. Default: 6.

```python
pn_gen_single()
```

Perlin noise single coordinate query.

```python
pn_gen_rect()
```