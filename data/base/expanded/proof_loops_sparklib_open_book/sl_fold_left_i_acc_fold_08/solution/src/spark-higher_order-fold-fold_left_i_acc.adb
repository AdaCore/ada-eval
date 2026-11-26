--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold)
package body Fold_Left_I_Acc is

   ----------
   -- Fold --
   ----------

   function Fold (A : Array_Type; Init : Element_Out) return Acc_Array
   is separate;

end Fold_Left_I_Acc;
