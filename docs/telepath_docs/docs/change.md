# Changelog

## 0.8.1.14
- added `explain` command
- added `compare` command
- predict can now take multiple agents in `agent_path`
- added `include_r` and `include_next` in `predict`
- added `col=` option to `replay_stats`

## 0.8.1.13
- improvements to the `predict` command
- clean up spec folders
- more consistent spec names
- added `--include_q` option, q value summaries, q value histograms
- added `--include_obs` to include observation values in prediction output

## 0.8.1.12

- lots of additions and improvements to [prep](prep.md) command scripting

## 0.8.1.11

- added [prep](prep.md) command scripting
- simplified [prep](prep.md) command arguments, replay specs no longer needed, added prep_mode, renamed xform to prep throughout
- added [replay_stats](cmd.md) command

## 0.8.1.10

- integration script generator
- dedicated spec folder per project, deprecation of specs2/
- fixes to prep command
- documentation improvements
- docker script improvements

## 0.8.1.8

- documentation improvements
- prep command
- refactored integration tests
- coninuous integration
- telepath docker, s3 versioning

## 0.8.1.5

- new cleaner command line API
