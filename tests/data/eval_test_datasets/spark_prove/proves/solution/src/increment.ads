function Increment (X : Integer) return Integer
with Pre => X < Integer'Last, Post => Increment'Result = X + 1;
