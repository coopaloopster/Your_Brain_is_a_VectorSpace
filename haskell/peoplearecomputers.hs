{-# LANGUAGE OverloadedStrings #-}
import Data.List
import qualified Data.Map as Map
import Data.Aeson
import Control.Applicative
import Data.Functor
import Control.Monad
import qualified Data.ByteString.Lazy as BL
import System.Random

data TrainingData = TD
    { flower :: Map.Map String Float, training_data :: [[Float]] } deriving (Show)

instance FromJSON TrainingData where
    parseJSON (Object v) = TD <$>
                        v .: "flower" <*>
                        v .: "training_data"

mmult :: Num a => [[a]] -> [[a]] -> [[a]]
mmult a b = [ [ sum $ zipWith (*) ar bc | bc <- (transpose b) ] | ar <- a ]

random_array :: RandomGen g => Int -> Int -> g -> ([[Float]], g)
random_array x y gen = 
    let random_list = randoms gen :: [Float]
        (_, gen') = random gen
    in ([ [ random_list !! (b*x + a) | a <- [1..x] ] | b <- [1..y] ], gen')

makeRandomArrays :: RandomGen g => [[Int]] -> g -> ([[[Float]]], g)
makeRandomArrays xs gen =
    ( [ rand | arr <- xs, let x = head arr, let y = head $ tail arr, let (rand, _) = random_array x y gen ], gen)

main :: IO()
main = do
    matrix <- getStdRandom (makeRandomArrays [[3, 4], [4,1]])
    print matrix
