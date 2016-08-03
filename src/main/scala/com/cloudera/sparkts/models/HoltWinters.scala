/**
 * Copyright (c) 2015, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.sparkts.models

<<<<<<< HEAD
import org.apache.commons.math3.analysis.{MultivariateFunction}
import org.apache.commons.math3.optim.{InitialGuess, MaxEval, MaxIter, SimpleBounds}
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import org.apache.commons.math3.optim.nonlinear.scalar.{GoalType, ObjectiveFunction}
import org.apache.spark.mllib.linalg._

/**
 * TODO: add explanation
 */
object HoltWinters {
  def fitModel(ts: Vector, m: Int, method: String = "BOBYQA"): HoltWintersModel = {
    method match {
      //case "CG" => fitModelWithCG(ts)
      case "BOBYQA" => fitModelWithBOBYQA(ts, m)
      case _ => throw new UnsupportedOperationException("Currently only supports 'CG' and 'BOBYQA'")
    }
  }

  def fitModelWithBOBYQA(ts: Vector, m: Int): HoltWintersModel = {
    val optimizer = new BOBYQAOptimizer(5)
    val objectiveFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new HoltWintersModel(m, params(0), params(1), params(2)).sse(ts)
      }
    })
    // The starting guesses in R's stats:HoltWinters
    val initGuess = new InitialGuess(Array(0.3, 0.1, 0.1))
    val maxIter = new MaxIter(10000)
    val maxEval = new MaxEval(10000)
    val goal = GoalType.MINIMIZE
    val bounds = new SimpleBounds(Array(0.0, 0.0, 0.0), Array(1.0, 1.0, 1.0))
    val optimal = optimizer.optimize(objectiveFunction, goal, bounds, initGuess, maxIter, maxEval)
    val params = optimal.getPoint
    new HoltWintersModel(m, params(0), params(1), params(2))
=======
import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.spark.mllib.linalg._
import org.apache.commons.math3.optim.MaxIter
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.SimpleBounds
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.BOBYQAOptimizer
import org.apache.commons.math3.optim.InitialGuess
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType

/**
 * Triple exponential smoothing takes into account seasonal changes as well as trends.
 * Seasonality is deﬁned to be the tendency of time-series data to exhibit behavior that repeats
 * itself every L periods, much like any harmonic function.
 *
 * The Holt-Winters method is a popular and effective approach to forecasting seasonal time series
 *
 * See https://en.wikipedia.org/wiki/Exponential_smoothing#Triple_exponential_smoothing
 * for more information on Triple Exponential Smoothing
 * See https://www.otexts.org/fpp/7/5 and
 * https://stat.ethz.ch/R-manual/R-devel/library/stats/html/HoltWinters.html
 * for more information on Holt Winter Method.
 */
object HoltWinters {

  /**
   * Fit HoltWinter model to a given time series. Holt Winter Model has three parameters
   * level, trend and season component of time series.
   * We use BOBYQA optimizer which is used to calculate minimum of a function with
   * bounded constraints and without using derivatives.
   * See http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf for more details.
   *
   * @param ts Time Series for which we want to fit HoltWinter Model
   * @param period Seasonality of data i.e  period of time before behavior begins to repeat itself
   * @param modelType Two variations differ in the nature of the seasonal component.
   *  	Additive method is preferred when seasonal variations are roughly constant through the series,
   *  	Multiplicative method is preferred when the seasonal variations are changing
   *  	proportional to the level of the series.
   * @param method: Currently only BOBYQA is supported.
   */
  def fitModel(ts: Vector, period: Int, modelType: String = "additive", method: String = "BOBYQA")
  : HoltWintersModel = {
    method match {
      case "BOBYQA" => fitModelWithBOBYQA(ts, period, modelType)
      case _ => throw new UnsupportedOperationException("Currently only supports 'BOBYQA'")
    }
  }

  def fitModelWithBOBYQA(ts: Vector, period: Int, modelType:String): HoltWintersModel = {
    val optimizer = new BOBYQAOptimizer(7)
    val objectiveFunction = new ObjectiveFunction(new MultivariateFunction() {
      def value(params: Array[Double]): Double = {
        new HoltWintersModel(modelType, period, params(0), params(1), params(2)).sse(ts)
      }
    })

    // The starting guesses in R's stats:HoltWinters
    val initGuess = new InitialGuess(Array(0.3, 0.1, 0.1))
    val maxIter = new MaxIter(30000)
    val maxEval = new MaxEval(30000)
    val goal = GoalType.MINIMIZE
    val bounds = new SimpleBounds(Array(0.0, 0.0, 0.0), Array(1.0, 1.0, 1.0))
    val optimal = optimizer.optimize(objectiveFunction, goal, bounds,initGuess, maxIter, maxEval)
    val params = optimal.getPoint
    new HoltWintersModel(modelType, period, params(0), params(1), params(2))
>>>>>>> upstream/master
  }
}

class HoltWintersModel(
<<<<<<< HEAD
  val m: Int,
  val alpha: Double,
  val beta: Double,
  val gamma: Double) extends TimeSeriesModel {
  /**
   * Calculates sum of squared errors, used to estimate the alpha and beta parameters
=======
    val modelType: String,
    val period: Int,
    val alpha: Double,
    val beta: Double,
    val gamma: Double) extends TimeSeriesModel {

  if (!modelType.equalsIgnoreCase("additive") && !modelType.equalsIgnoreCase("multiplicative")) {
    throw new IllegalArgumentException("Invalid model type: " + modelType)
  }
  val additive = modelType.equalsIgnoreCase("additive")

  /**
   * Calculates sum of squared errors, used to estimate the alpha and beta parameters
   *
>>>>>>> upstream/master
   * @param ts A time series for which we want to calculate the SSE, given the current parameters
   * @return SSE
   */
  def sse(ts: Vector): Double = {
    val n = ts.size
    val smoothed = new DenseVector(Array.fill(n)(0.0))
    addTimeDependentEffects(ts, smoothed)

    var error = 0.0
    var sqrErrors = 0.0
<<<<<<< HEAD
    var i = 0

    while (i <  n) {
      error = ts(i) - smoothed(i)
      sqrErrors += error * error
      i += 1
    }
=======

    // We predict only from period by using the first period - 1 elements.
    for(i <- period to (n - 1)) {
      error = ts(i) - smoothed(i)
      sqrErrors += error * error
    }

>>>>>>> upstream/master
    sqrErrors
  }

  /**
   * {@inheritDoc}
   */
  override def removeTimeDependentEffects(ts: Vector, dest: Vector = null): Vector = {
    throw new UnsupportedOperationException("not yet implemented")
  }

  /**
   * {@inheritDoc}
   */
  override def addTimeDependentEffects(ts: Vector, dest: Vector): Vector = {
    val destArr = dest.toArray
    val fitted = getHoltWintersComponents(ts)._1
<<<<<<< HEAD
    for (i <- 0 until dest.size) {
=======
    for (i <- 0 to (dest.size - 1)) {
>>>>>>> upstream/master
      destArr(i) = fitted(i)
    }
    dest
  }

<<<<<<< HEAD

  /**
   * TODO, explain
   * @param ts
   * @param dest
   */
  def forecast(ts: Vector, dest: Vector) = {
    val destArr = dest.toArray
    val (fitted, level, trend, season) = getHoltWintersComponents(ts)
    val n = ts.size
    var si = 0
    val levelVal = level(n - 1)
    val trendVal = trend(n - 1)

    for (i <- 0 until dest.size) {
      // if in sample, fitted, else forecasted
      if (i > n - 1) {
          si = if ((i - n) % m == 0) n else (n - m) + ((i - n) % m)
          destArr(i) = levelVal + (i - n + 1) * trendVal + season(si - 1)
        } else {
          destArr(i) = fitted(i)
        }
      si += 1
=======
  /**
   * Final prediction Value is sum of level trend and season
   * But in R's stats:HoltWinters additional weight is given for trend
   *
   * @param ts
   * @param dest
   */
  def forecast(ts: Vector, dest: Vector): Vector = {
    val destArr = dest.toArray
    val (_, level, trend, season) = getHoltWintersComponents(ts)
    val n = ts.size

    val finalLevel = level(n - period)
    val finalTrend = trend(n - period)
    val finalSeason = new Array[Double](period)

    for (i <- 0 until period) {
      finalSeason(i) = season(i + n - period)
    }

    for (i <- 0 until dest.size) {
      destArr(i) = if (additive) {
        (finalLevel + (i + 1) * finalTrend) + finalSeason(i % period)
      } else {
        (finalLevel + (i + 1) * finalTrend) * finalSeason(i % period)
      }
>>>>>>> upstream/master
    }
    dest
  }

  /**
<<<<<<< HEAD
   * TODO: explain
   * 3 components, level, trend, seasonality
=======
   * Start from the intial parameters and then iterate to find the final parameters
   * using the equations of HoltWinter Method.
   * See https://www.otexts.org/fpp/7/5 and
   * https://stat.ethz.ch/R-manual/R-devel/library/stats/html/HoltWinters.html
   * for more information on Holt Winter Method equations.
   *
   * @param ts A time series for which we want the HoltWinter parameters level,trend and season.
   * @return (level trend season). Final vectors of level trend and season are returned.
>>>>>>> upstream/master
   */
  def getHoltWintersComponents(ts: Vector): (Vector, Vector, Vector, Vector) = {
    val n = ts.size
    require(n >= 2, "Requires length of at least 2")

    val dest = new Array[Double](n)
<<<<<<< HEAD
    val level = new Array[Double](n)
    val trend = new Array[Double](n)

    // http://robjhyndman.com/hyndsight/hw-initialization/
    // We follow the simple method (1998), and leave as TODO
    // to initalize using the 2008 suggestion
    val (initLevel, initTrend, season) = initHoltWintersSimple(ts)

    var prevTrend = initLevel
    var prevLevel = initTrend
    var si = 0
    dest(0) = initLevel + initTrend + season(0)

    for (i <- 0 until ts.size - 1) {
      si = if (i >= m) i - m else i
      level(i) = alpha * (ts(i) - season(si)) + (1 - alpha) * (prevLevel + prevTrend)
      trend(i) =  beta * (level(i) - prevLevel) + (1 - beta) * prevTrend
      // We'll stick to this variant, so that we can impose constraints on parameters
      // easily with BOBYQA
      if (i >= m ) { // only update seasonality after the first m periods
        season(i) = gamma * (ts(i) - level(i)) + (1 - gamma) * season(si)
      }
      prevLevel = level(i)
      prevTrend = trend(i)
      dest(i + 1) = level(i) + trend(i) + season(si + 1)
=======

    val level = new Array[Double](n)
    val trend = new Array[Double](n)
    val season = new Array[Double](n)

    val (initLevel, initTrend, initSeason) = initHoltWinters(ts)
    level(0) = initLevel
    trend(0) = initTrend
    for (i <- 0 until initSeason.size){
      season(i) = initSeason(i)
    }

    for (i <- 0 to (n - period - 1)) {
      dest(i + period) = level(i) + trend(i)

      // Add the seasonal factor for additive and multiply for multiplicative model.
      if (additive) {
        dest(i + period) += season(i)
      } else {
        dest(i + period) *= season(i)
      }

      val levelWeight = if (additive) {
        ts(i + period) - season(i)
      } else {
        ts(i + period) / season(i)
      }

      level(i + 1) = alpha * levelWeight + (1 - alpha) * (level(i) + trend(i))

      trend(i + 1) = beta * (level(i + 1) - level(i)) + (1 - beta) * trend(i)

      val seasonWeight = if (additive) {
        ts(i + period) - level(i + 1)
      } else {
        ts(i + period) / level(i + 1)
      }
      season(i + period) = gamma * seasonWeight + (1 - gamma) * season(i)
>>>>>>> upstream/master
    }

    (Vectors.dense(dest), Vectors.dense(level), Vectors.dense(trend), Vectors.dense(season))
  }

<<<<<<< HEAD
  //TODO: add check for length...bad method for short/noisy series as per source
  //TODO: implemente alternative start described in Hyndman 2008
  def initHoltWintersSimple(ts: Vector): (Double, Double, Array[Double]) = {
    val arrTs = ts.toArray
    val lm = arrTs.take(m).sum / m //average value for first m obs
    val bm = arrTs.take(m * 2).splitAt(m).zipped.map { case (prevx, x) =>
        x - prevx
      }.sum / (m * m)
    val siPrelim = arrTs.take(m).map(_ - lm)
    //first m periods initialized to seasonal values, rest zeroed out
    val si = siPrelim ++ Array.fill(arrTs.length - m)(0.0)
    (lm, bm, si)
=======
  def getKernel(): (Array[Double]) = {
    if (period % 2 == 0){
      val kernel = Array.fill(period + 1)(1.0 / period)
      kernel(0) = 0.5 / period
      kernel(period) = 0.5 / period
      kernel
    } else {
      Array.fill(period)(1.0 / period)
    }
  }

  /**
   * Function to calculate the Weighted moving average/convolution using above kernel/weights
   * for input data.
   * See http://robjhyndman.com/papers/movingaverage.pdf for more information
   * @param inData Series on which you want to do moving average
   * @param kernel Weight vector for weighted moving average
   */
  def convolve(inData: Array[Double], kernel: Array[Double]): (Array[Double]) = {
    val kernelSize = kernel.size
    val dataSize = inData.size

    val outData = new Array[Double](dataSize - kernelSize + 1)

    var end = 0
    while (end <= (dataSize - kernelSize)) {
      var sum = 0.0
      for (i <- 0 until kernelSize) {
        sum += kernel(i) * inData(end + i)
      }

      outData(end) = sum
      end += 1
    }

    outData
  }

  /**
   * Function to get the initial level, trend and season using method suggested in
   * http://robjhyndman.com/hyndsight/hw-initialization/
   * @param ts
   */
  def initHoltWinters(ts: Vector): (Double, Double, Array[Double]) = {
    val arrTs = ts.toArray

    // Decompose a window of time series into level trend and seasonal using convolution
    val kernel = getKernel()
    val kernelSize = kernel.size
    val trend = convolve(arrTs.take(period * 2), kernel)

    // Remove the trend from time series. Subtract for additive and divide for multiplicative
    val n = (kernelSize -1) / 2
    val removeTrend = arrTs.take(period * 2).zip(
      Array.fill(n)(0.0) ++ trend ++ Array.fill(n)(0.0)).map{
      case (a, t) =>
        if (t != 0){
          if (additive) {
            (a - t)
          } else {
            (a / t)
          }
        }  else{
          0
        }
    }

    // seasonal mean is sum of mean of all season values of that period
    val seasonalMean = removeTrend.splitAt(period).zipped.map { case (prevx, x) =>
      if (prevx == 0 || x == 0) (x + prevx) else (x + prevx) / 2
    }

    val meanOfFigures = seasonalMean.sum / period

    // The seasonal mean is then centered and removed to get season.
    // Subtract for additive and divide for multiplicative.
    val initSeason = if (additive) {
      seasonalMean.map(_ - meanOfFigures )
    } else {
      seasonalMean.map(_ / meanOfFigures )
    }

    // Do Simple Linear Regression to find the initial level and trend
    val indices = 1 to trend.size
    val xbar = (indices.sum: Double) / indices.size
    val ybar = trend.sum / trend.size

    val xxbar = indices.map( x => (x - xbar) * (x - xbar) ).sum
    val xybar = indices.zip(trend).map {
      case (x, y) => (x - xbar) * (y - ybar)
    }.sum

    val initTrend = xybar / xxbar
    val initLevel = ybar - (initTrend * xbar)

    (initLevel, initTrend, initSeason)
>>>>>>> upstream/master
  }
}
