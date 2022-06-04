# Проекты Data Science

В репозитарии собраны проекты, выполненные в учебных целях. Проекты представлены в формате тетрадок iPython. Проекты были проверены наставниками Яндекс.Практикум.

_Примечание: датасеты, использованные в проектах, не могут быть предоставлены из-за юридических ограничений на использование._

## Содержание

- ### Машинное обучение
	- [Модель для определения стоимости автомобилей](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/autoprice_predictions.ipynb): предсказание стоимости автомобиля на основании технических характеристик и комплектации. Предсказывает лучшую цену продажи с использованием машинного обучения.
	- [Рекомендация тарифов](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/tariff_recomendations.ipynb): на основании данных о поведении клиентов сотового оператора, построена модель для задачи классификации, которая выберет подходящий тариф для пользователя: построены пять различных моделей со значением accuracy выше 0.75.
	- [Отток клиентов](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/churn_predictions.ipynb): предсказание расторжения договора клиентом с банком средствами машинного обучения. Борьба с дисбалансом классов.
	- [Температура стали](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/steel_temperature_predictions.ipynb): предсказание температуры расплавленной стали.
	- [Число такси на следующий час](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/taxi_predictions.ipynb): прогноз количества заказов такси на следующий час.
	- [Анализ прибыли и рисков](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/profit_and_risks_analysis.ipynb): на основании данных о пробах нефти в трёх регионах, качестве нефти и объемов запасов построена модель машинного обучения для определения региона, где добыча принесёт наибольшую прибыль. Линейная регрессия, статистические тесты, bootstrap.
	- [Коэффициент восстановления золота из золотосодержащей руды](https://github.com/ikazman/praktikum-ds-projects/blob/main/machine_learning/gold_recovery.ipynb): на основании данных с параметрами добычи и очистки модель предсказывает коэффициент восстановления золота из золотосодержащей руды. Исследовательский анализ данных, предобработка, машинное обучение.
	
	_Инструменты: scikit-learn, LightGBM, Catboost, Pandas, Seaborn, Matplotlib_

- ### Процессинг естественного языка (NLP)
	- [Классификация комментариев на позитивные и негативные](https://github.com/ikazman/praktikum-ds-projects/blob/main/natural_language_processing/positive_analysis.ipynb): инструмент для поиска токсичных комментариев. Очистка и векторизация текстов.
	
	_Инструменты: NLTK, scikit-learn_
	
- ### Компьютерное зрение
	- [Модель для автоматического определения возраста](https://github.com/ikazman/praktikum-ds-projects/blob/main/computer_vision/age_detections.ipynb): решение задачи регрессии при обработке изображений.
	
	_Инструменты: Tensorflow_

- ### Анализ данных и визуализация
	- [Исследование тарифов сотового оператора](https://github.com/ikazman/praktikum-ds-projects/blob/main/analysis_and_vizualizations/cellphones_tariffs.ipynb): анализ поведения клиентов в зависимости от избранного тарифа. Исследовательский анализ данных, предобработка данных, статистические тесты, визуализация данных.
	- [Исследование продаж компьютерных игр](https://github.com/ikazman/praktikum-ds-projects/blob/main/analysis_and_vizualizations/games_sales.ipynb): анализ рынка компьютерных игр, выявление закономерностей популярности игр. Предобработка данных, исследовательский анализ данных, визуализация, статистические тесты, предсказание оценок критиков с помощью KNN.
	- [Исследование надёжности заёмщиков](https://github.com/ikazman/praktikum-ds-projects/blob/main/analysis_and_vizualizations/borrowers_realiability.ipynb): исследование влияния целей кредита, семейного положения, количества детей клиента на факт погашения кредита в срок.
	- [Исследование объявлений о продаже квартир](https://github.com/ikazman/praktikum-ds-projects/blob/main/analysis_and_vizualizations/adverts_of_apartments.ipynb): исследование влияния параметров квартиры (местоположение, характеристики) на рыночную стоимость квартиры. На основе данных обучена модель, предсказывающая стоимость квартиры.
	- [Исследование предпочтений пользователей, покупающих билеты на разные направления](https://github.com/ikazman/praktikum-ds-projects/blob/main/analysis_and_vizualizations/aviatickets_buyers_analysis.ipynb): анализ спроса пассажиров на рейсы в города, где проходят крупнейшие культурные фестивали.

	_Инструменты: Pandas, Numpy, Scipy, Statsmodel, Scikit-learn, Seaborn, Plotly, Matplotlib, Pymystem3_

- ### Иные проекты:
	- [Шифрование данных](https://github.com/ikazman/praktikum-ds-projects/blob/main/others/data_encryption.ipynb): очень простое преобразование данных с помощью Numpy с целью шифрования с обоснованием метода.

	_Инструменты: Pandas, Numpy_
