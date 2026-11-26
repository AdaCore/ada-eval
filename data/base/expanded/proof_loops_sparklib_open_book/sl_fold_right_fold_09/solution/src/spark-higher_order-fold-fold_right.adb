--
--  Copyright (C) 2018-2026, AdaCore
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

separate (SPARK.Higher_Order.Fold)
package body Fold_Right is

   ----------
   -- Fold --
   ----------

   function Fold (A : Array_Type; Init : Element_Out) return Element_Out
   is separate;
end Fold_Right;
