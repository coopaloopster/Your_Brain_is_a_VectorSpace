import Data.List

main = print (mmult [[1,2], [3,4]] [[5,6], [1,3]])

mmult :: Num a => [[a]] -> [[a]] -> [[a]]
mmult a b = [ [ sum $ zipWith (*) ar bc | bc <- (transpose b) ] | ar <- a ]
