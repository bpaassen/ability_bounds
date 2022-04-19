# Faster Confidence Intervals for Item Response Theory via an Approximate Likelihood

Copyright (C) 2021-2022  
Benjamin Paaßen  
German Research Center for Artificial Intelligence

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.



This is the accompanying source code for the EDM 2022 poster 'Faster Confidence Intervals for Item Response Theory via an Approximate Likelihood'. If you use this implementation in academic work, please cite the paper

* Paaßen, B., Göpfert, C., & Pinkwart, N. (2022). Faster Confidence Intervals for Item Response Theory via an Approximate Likelihood. In: Cristea, A., Brown, C., Mitrovic, T., & Bosch, N. (Eds.). Proceedings of the 15th International Conference on Educational Datamining (EDM 2022). accepted.

```bibtex
@inproceedings{Paassen2022EDM,
    author       = {Paaßen, Benjamin and Göpfert, Christina and Pinkwart, Niels},
    title        = {Faster Confidence Intervals for Item Response Theory via an Approximate Likelihood},
    booktitle    = {{Proceedings of the 14th International Conference on Educational Data Mining (EDM 2022)}},
    date         = {2022-07-24},
    year         = {2022},
    venue        = {Durham, UK},
    editor       = {Cristea, Alexandra I. and Brown, Chris and Mitrovic, Tanja and Bosch, Nigel},
    note         = {accepted}
}
```

The reference implementation for all confidence bound methods can be found in `ability_bounds.py`. The experimental source code in the notebook `synthetic_experiments.ipynb`.

The source code in this repository depends on [numpy](https://numpy.org/) and [scipy](https://scipy.org/).
