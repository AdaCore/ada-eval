--  The C binding's body is not part of the extraction: every operation the
--  loop reaches is imported with its upstream contract.  The body exists only
--  because the spec declares an abstract state that the loop writes, and a
--  package of imported specs with no body has that state rejected as never
--  initialized (extraction recipe, "a package of Import'ed specs has no
--  body").  SPARK_Mode => Off keeps it outside the perimeter.

package body Sqlite_Vec_Spark
  with SPARK_Mode => Off
is
end Sqlite_Vec_Spark;
