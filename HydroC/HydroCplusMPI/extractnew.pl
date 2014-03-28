#!/usr/bin/env perl
while ( <> ) {
   @a=split(" ");
   $elaps=$a[4];
   $threads=$a[10];
   print $threads . " " . $elaps ."\n";
}
