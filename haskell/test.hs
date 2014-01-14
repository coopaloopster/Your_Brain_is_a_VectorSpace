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

random_array =
    let random_list = randoms getStdGen :: [Float]
    in [[rand | rand <- random_list, a <- [0..3]] | b <- [0..4]]

main :: IO()
main = do
    a <- BL.readFile "../data.json"
    (Just b) <- return (decode a :: Maybe TrainingData)
    print b
