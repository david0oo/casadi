// NOLINT(legal/copyright)
// SYMBOL "file_slurp"
template <typename T1>
int casadi_file_slurp(const char* fname, casadi_int n, T1* x) {
  casadi_int i;
  FILE *fp;
  fp = fopen(fname, "r");
  if (!fp) return 1;
  for (i=0;i<n;++i) {
    // C-VERBOSE
    constexpr char SCANF_REAL[4] = sizeof(T1)==sizeof(float) ? "%g\0" : "%lg";
    if (fscanf(fp, SCANF_REAL, x++)<=0) return 2;
  }
  fclose(fp);
  return 0;
}
