package Search_Array is

   type Array_Of_Positives is array (Natural range <>) of Positive;

   Not_Found : exception;

   type Search_Result (Found : Boolean := False) is record
      case Found is
         when True =>
            Content : Integer;

         when False =>
            null;
      end case;
   end record;

   procedure Search_Array
     (A : Array_Of_Positives; E : Positive; Result : out Search_Result);

end Search_Array;
