Before: Coarse inference for entire tuple
Avg CPU for each tuple 
Perf one issue --fixed
Changes needed:
Granular for each .predict
CPU samples for each inference should not be averaged
rather it would keep them in sereis format