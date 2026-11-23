# Replay Prep Scripts

With the `prep` command, Telepath can transform Replay data to make it suitable for input.


```
telepath prep \
     --project_spec=telepath_settings.yaml \
     --in_replay=replay_rawtrain.csv \
     --out_replay=replay.csv \
     --prep_spec=replay_prep_ident.yaml \
     --prep_mode=train \
     --debug
```

Transformation steps are listed in the `prep_scripts` section of the prep spec file, in the sequence they are to be executed.

For example:
```
telepath_spec: 2
spec_type: replay_prep
content:
  -
    prep_mode: all
    script:
      - sort account_id day
      - fillna0 recent_weekly_email_count recent_weekly_event_count email_delivered_at day_of_week
```
These preparation scripts are essentially a domain specific language, and can be as simple or elaborate as needed.

Representing these scripts in data instead of code, allows for comparison of multiple variations of preparation scripts, without code switches or code changes.

## prep_mode

Steps are grouped into scripts which are executed in different prep_modes, depending on whether the data is meant to predict or train.

Input data used prediction must have the same properties as the training data.   Common preparation steps are marked `prep_mode: all`.  For example, removing nulls with `fillna0`.

Prep scripts with commands specific to training data (for example, shuffling with `sort_rand`) are marked `prep_mode: train`.

Prep scripts with commands specific to prediction data (for example, filtering out training rows) are marked `prep_mode: predict`.

The `prep_mode` argument on the prep command line, controls which of these sections is executed: use either `--prep_mode=train` or `--prep_mode=predict`


## prep_script commands

### Modify/append individual columns

- `bool f1 f2 f3` : Convert true/false column to 1/0 ints.  Does `fillna0` first.
- `const f val` : Create constant column f with numeric value val.
- `f16 f1 f2 f3` :  Convert columns to float16 type
- `f32 f1 f2 f3` :  Convert columns to float32 type
- `fillna0 f1 f2 f3` :  Replace any NaNs in these cols with 0.0
- `i32 f1 f2 f3` :  Convert columns to int32 type
- `log f1 f2 f3` :  Append log10(x) column with suffix '_log'
- `logp1 f1 f2 f3` :  Append log10(x+1.0) column with suffix '_log' (+1.0 to avoid overflow if x==0.0)

### Sorting

- `sort f1 f2 f3` :  Sort by a list of fields.
- `sort_rand` : Randomly shuffle all rows.

### Modify column list

- `rm f1 f2 f3` :  Remove columns
- `sel f1 f2 f3` :  Select only these columns
- `ren fld_before fld_after` :  Rename a column

### Indicator fields
- `nonzero f1 f2 f3` : If field is zero, keep at 0, otherwise 1
- `nanz f1 f2 f3` : If field is NaN, set to 0, otherwise 1
- `match_2f dest_fld comp_fld1 comp_fld2` : Append 0/1 indicator field dest_fld showing if 2 other fields match; if comp_fld1==comp_fld2, then set dest_fld=1, otherwise 0

### Vertical operations
- `shift offset f1 f2 f3` : append vertically shifted version of fields, with suffix like `_shn1` or `_shp2` (ie. "shift negative 1", "shift positive 2")
- `roll operator offset f1 f2 f3` : append rolling aggregate of fields, with suffix like `..._roll_sum_p2`

### Expression operations
- `eval fld=<eval_expression>` : Construct a new field based on expression, eg: `eval newfield = df.obs2*2.5 + df.obs1*10.0`, see [pandas eval](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html)  
- `query <query_expression>` : Filter down to rows that match query, , eg: `query account_id > 10 & account_id < 100`, see [pandas query](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html)

### Other operators
- `noop` : No-operation. Do nothing. A placeholder.


## Common prep patterns

- group common preparation steps into a few scripts:
    - start with a `prep_mode: all` script to do cleaning and featurization
    - add a `prep_mode: predict` script to filter out training data
    - add a `prep_mode: train` script to:
        - filter out rows with no next-step data
        - filter out rows that dont have the future data they require
        - randomly shuffle with `sort_rand`
- tidying
    - reduce memory requirements by converting fields to `f16` and `i32` formats where possible
    - clean NaNs using `fillna0` and `nanz`
    - make binary indicator fields using `nonzero` and `nanz`
- shifting
    - `sort` by account_id and day, to allow for sensible `shift` and `roll` results in the rest of the script
    - use `shift -1 ...` to create "next observation" fields where they are missing
    - create "last row" binary indicator field `match_nx_aid`, using `shift -1 account_id` and `match_2f match_nx_aid account_id account_id_shn1 ` to allow special treatment for last row for each account
    - zero-out invalid "next observations" for "last row", my multiplying with indicator field `match_nx_aid` using 
`mult_f match_nx_aid ...`
- adjust the final set of fields with `rm` and `sel`


## Debugging

The optional `--debug` flag show frame summaries at every step of the prep script, to help determine what went wrong.
