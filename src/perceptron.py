"""Perceptron binário implementado com NumPy, sem scikit-learn."""
from __future__ import annotations

import numpy as np


class Perceptron:
    """Classificador linear para rótulos -1 e +1.

    A atualização de Rosenblatt ocorre somente quando há erro:
    w <- w + taxa * y * x; b <- b + taxa * y.
    """

    def __init__(self, taxa_aprendizado: float = 0.1, epocas: int = 100):
        if not np.isfinite(taxa_aprendizado) or taxa_aprendizado <= 0:
            raise ValueError("taxa_aprendizado deve ser um número positivo e finito")
        if isinstance(epocas, bool) or not isinstance(epocas, int) or epocas < 1:
            raise ValueError("epocas deve ser inteiro positivo")
        self.taxa_aprendizado = float(taxa_aprendizado)
        self.epocas = epocas
        self.pesos_: np.ndarray | None = None
        self.vies_: float = 0.0
        self.erros_por_epoca_: list[int] = []

    @staticmethod
    def _validar_x(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.all(np.isfinite(X)):
            raise ValueError("X deve ser matriz 2D não vazia, com valores finitos")
        return X

    def ajustar(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = self._validar_x(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or not np.all(np.isin(y, [-1, 1])):
            raise ValueError("y deve ter um rótulo -1 ou +1 por amostra")
        self.pesos_ = np.zeros(X.shape[1], dtype=float)
        self.vies_ = 0.0
        self.erros_por_epoca_ = []
        for _ in range(self.epocas):
            erros = 0
            for xi, yi in zip(X, y):
                if yi * (np.dot(xi, self.pesos_) + self.vies_) <= 0:
                    self.pesos_ += self.taxa_aprendizado * yi * xi
                    self.vies_ += self.taxa_aprendizado * yi
                    erros += 1
            self.erros_por_epoca_.append(erros)
            if erros == 0:
                break
        return self

    def pontuacao_decisao(self, X: np.ndarray) -> np.ndarray:
        if self.pesos_ is None:
            raise RuntimeError("Treine o modelo usando ajustar() antes de prever")
        X = self._validar_x(X)
        if X.shape[1] != len(self.pesos_):
            raise ValueError("Quantidade de atributos incompatível")
        return X @ self.pesos_ + self.vies_

    def prever(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.pontuacao_decisao(X) >= 0, 1, -1)

    def acuracia(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.prever(X)
        y = np.asarray(y)
        if y.shape != pred.shape or not np.all(np.isin(y, [-1, 1])):
            raise ValueError("y deve ter um rótulo -1 ou +1 por amostra")
        return float(np.mean(pred == y))
