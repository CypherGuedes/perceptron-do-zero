import unittest
import numpy as np
from src.perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    def test_porta_and(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, -1, -1, 1])
        modelo = Perceptron(taxa_aprendizado=0.1, epocas=100).ajustar(X, y)
        np.testing.assert_array_equal(modelo.prever(X), y)
        self.assertEqual(modelo.acuracia(X, y), 1.0)
        self.assertEqual(modelo.erros_por_epoca_[-1], 0)

    def test_porta_or(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, 1, 1, 1])
        modelo = Perceptron().ajustar(X, y)
        np.testing.assert_array_equal(modelo.prever(X), y)

    def test_previsao_antes_do_treino(self):
        with self.assertRaises(RuntimeError):
            Perceptron().prever([[1, 2]])

    def test_parametros_invalidos(self):
        with self.assertRaises(ValueError):
            Perceptron(taxa_aprendizado=0)
        with self.assertRaises(ValueError):
            Perceptron(epocas=0)

    def test_dados_invalidos(self):
        with self.assertRaises(ValueError):
            Perceptron().ajustar([[0], [1]], [0, 1])
        with self.assertRaises(ValueError):
            Perceptron().ajustar([[0], [1]], [1])

    def test_atributos_incompativeis(self):
        modelo = Perceptron().ajustar([[0, 0], [1, 1]], [-1, 1])
        with self.assertRaises(ValueError):
            modelo.prever([[1, 1, 1]])


if __name__ == '__main__':
    unittest.main()
