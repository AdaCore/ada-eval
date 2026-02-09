with Types; use Types;

package Placeholder is

   function Has_Close_Elements
     (Numbers : Float_Array; Threshold : Float) return Boolean;
   --  Check if in given Array of numbers, are any two numbers closer to each
   --  other than the given threshold.
   --  >>> Has_Close_Elements ([1.0, 2.0, 3.0], 0.5)
   --  False
   --  >>> Has_Close_Elements ([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
   --  True

end Placeholder;
