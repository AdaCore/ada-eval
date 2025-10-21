package Show_Permutation is

   type Permutation is array (Positive range <>) of Positive;

   procedure Swap (A : in out Permutation; I, J : Positive)
   with Pre => I in A'Range and then J in A'Range;

   procedure Init (A : out Permutation);

   function Cyclic_Permutation (N : Natural) return Permutation;

end Show_Permutation;
