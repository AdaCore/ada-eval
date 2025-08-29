function Increment (X : Integer) return Integer
with Post => Increment'Result = X + 2;
