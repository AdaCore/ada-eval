package Show_Permutation is

   type Permutation is array (Positive range <>) of Positive;

   procedure Swap (A : in out Permutation; I, J : Positive);

   procedure Init (A : in out Permutation);

   function Cyclic_Permutation (N : Natural) return Permutation;

end Show_Permutation;
