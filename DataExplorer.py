class DataExplorer:


    def __init__(self):
        self.final_report = None
        self.best_estimator = []
        self.predictions = []

    def histogram(self, data, n_bins, range_start, range_end, grid, cumulative=False, x_label = '', y_label = '', title = ''):

        """
        Простая гистограмма

        Пример:
        histogram(df, 100, 0, 150, True, 'Количество иксов', 'Количество игриков', 'Заголовок')

        data - датасет
        n_bins - количество корзин
        range_start - минимальный икс для корзины
        range_end - максимальный икс для корзины
        grid - рисовать сетку или нет (False / True)


        histogram(data, n_bins, range_start, range_end, grid, x_label = "", y_label = "", title = "")
        """

        # Создаем объект - график
        _, ax = plt.subplots()

        # Задаем параметры
        ax.hist(data, bins = n_bins, range = (range_start, range_end), cumulative = cumulative, color = '#4169E1')

        # Добавляем сетку
        if grid == True:
            ax.grid(color='grey', linestyle='-', linewidth=0.5)
        else:
            pass

        # Добавляем медиану, среднее и квартили
        ax.axvline(data.median(),linestyle = '--', color = '#FF1493', label = 'median')
        ax.axvline(data.mean(),linestyle = '--', color = 'orange', label = 'mean')
        ax.axvline(data.quantile(0.1),linestyle = '--', color = 'yellow', label = '1%')
        ax.axvline(data.quantile(0.99),linestyle = '--', color = 'yellow', label = '99%')
        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)


    def scatterplot(self, x_data, y_data, x_label='', y_label='', title='', color = 'r', yscale_log=False, figsize = (8, 6)):

        """
        Простая диаграмма рассеивания

        Пример:
        scatterplot(df.real_target, df.predicted_target, x_label='Предсказанное моделью', y_label='Настоящий показатель', title='Диаграмма рассеивания')

        x_data - определяем иксы
        y_data - определяем игрики

        scatterplot(self, x_data, y_data, x_label='', y_label='', title='', color = 'r', yscale_log=False, figsize = (8, 6)):
        """

        # Создаем объект - график
        _, ax = plt.subplots(figsize = (8, 6))

        # Задаем параметры для графика, определяем размер (s), цвет и прозрачность точек на графике
        ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

        if yscale_log == True:
            ax.set_yscale('log')

        # Создаем описание осей и заголовок для графика
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)


    def overlaid_histogram(self, data1, data2, n_bins = 0, data1_name='', data1_color='#539caf', data2_name='', data2_color='#7663b0', x_label='', y_label='', title=''):

        """
        Гистогорамма для двух выборок с одинаковыми границами бинов
        Пример:
        overlaid_histogram(df.one, df.two, n_bins = 80, data1_name='Первый датасет', data2_name='Второй датасет', x_label='Признак', y_label='Частота', title='Гистограмма')
        data1 - первый датасет
        data2 - второй датасет
        n_bins - количество корзин
        """


        # Устанавливаем границы для корзин так чтобы оба распределения на графике были соотносимы
        max_nbins = 10
        data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
        binwidth = (data_range[1] - data_range[0]) / max_nbins


        if n_bins == 0:
            bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
        else:
            bins = n_bins

        # Рисуем график
        _, ax = plt.subplots(figsize=(10,8))
        ax.hist(data1, bins = bins, color = data1_color, alpha = 0.65, label = data1_name)
        ax.hist(data2, bins = bins, color = data2_color, alpha = 0.65, label = data2_name)

        ax.axvline(data1.mean(),linestyle = '--', color = 'lime', label = 'mean for data 1')

        ax.axvline(data2.mean(),linestyle = '--', color = 'coral', label = 'mean for data 2')

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_title(title)
        ax.legend(loc = 'best')


    def corr_diagram(self, x):

        """
        Диаграмма корреляции
        Пример:
        corr_diagram(self, data):
        """

        plt.figure(figsize=(12,10), dpi= 80)
        sns.heatmap(x.corr(), xticklabels=x.corr().columns, yticklabels=x.corr().columns, cmap='RdYlGn', center=0, annot=True)
        plt.title('Диаграмма корреляции', fontsize=22)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()


    def highlight_max(self, data, color='#00FF00'):

        """
        Подсвечивает максимумы в Series или DataFrame

        highlight_max(data)
        """

        attr = 'background-color: {}'.format(color)
        #remove % and cast to float
        data = data.replace('%','', regex=True).astype(float)
        data[data == 1] = None
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = (data == data.abs().max()) & (data !=1)
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = (data == data.abs().max()) & (data !=1)
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)


    def highlight_sorted_corr(self, data, threshold, color='#00FF00'):

        """
        Подсвечивает значения выше определенного порога в Series или DataFrame (для одного столбца)

        highlight_sorted_corr(data, threshold)
        """

        attr = 'background-color: {}'.format(color)
        #remove % and cast to float
        data = data.replace('%','', regex=True).astype(float)
        data[data == 1] = None
        if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
            is_max = (data > threshold) & (data !=1)
            return [attr if v else '' for v in is_max]
        else:  # from .apply(axis=None)
            is_max = (data == data.abs().max()) & (data !=1)
            return pd.DataFrame(np.where(is_max, attr, ''),
                                index=data.index, columns=data.columns)


    def lineplot(self, x_data, y_data, x_label="", y_label="", title=""):


        """
        Простой линейный график

        Пример:
        lineplot(df.some_x, df.some_y, x_label='Обозначения икс', y_label='Обозначения игрик', title='Заголовок')
        """


        # Создаем объект - график
        _, ax = plt.subplots(figsize=(8, 6))

        # Задаем параметры для линии: ширину (lw), цвет и прозрачность (alpha)
        ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)

        # Даем имена осям и заголовок для графика
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)


    def double_lineplot(self, x_data_1, y_data_1, x_data_2, y_data_2, x_label='', y_label='', title='', label_one='', label_two=''):

        """
        Простой двойной линейный график

        Пример:
        double_lineplot(df.some_x_1, df.some_y_1, df.some_x_2, df.some_y_2, x_label='Обозначения икс', y_label='Обозначения игрик', title='Заголовок', label_one='Линия 1', label_two='Линия 2'):
        """


        # Создаем объект - график
        _, ax = plt.subplots(figsize=(8, 6))

        # Задаем параметры для линии: ширину (lw), цвет и прозрачность (alpha)
        ax.plot(x_data_1, y_data_1, lw = 2, color = '#6400e4', alpha = 1, label = label_one)
        ax.plot(x_data_2, y_data_2, lw = 2, color = '#ffc740', alpha = 1, label = label_two)

        # Даем имена осям и заголовок для графика
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc = 'best')


    def hexbin(self, data, x, y):

        """
        Простой график с сотами

        Пример:
        hexbin(df, df.true_target, df.predicted_target)
        """

        data.plot(x = x, y = y, kind='hexbin', gridsize=20, figsize=(8, 6), sharex=False, grid=True)


    def bar_plotter(self, data):

        """
        Простой столбчатый график

        Пример:
        bar_plotter(data):
        """

        data.plot.bar(rot=0, figsize = (16, 5))


    def categorical_counter_plot(self, data, column, x = '', y = ''):

        """
        График для подсчета значений по категориям

        Пример:
        categorical_counter_plot(df, 'predicted', x = '10', y = '6'):
        """

        if x == '' or y == '':
            plt.rcParams["figure.figsize"] = (15, 10)
        else:
            plt.rcParams["figure.figsize"] = (x, y)

        order = data[column].value_counts().index

        ax = sns.countplot(data[column], order = order)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)

        plt.xticks(rotation=90)


    def sns_scatterplot(self, data, x='', y='', hue='', size='', palette=''):

        """
        Диаграмма рассеивания seaborn

        Пример:
        sns_scatterplot(platform_scores_wo, 'user_score', 'total_sales', 'critic_score', 'year_of_release', True)
        """

        sns.set(style="whitegrid")

        f, ax = plt.subplots(figsize=(15, 10))

        if palette == True:
            sns.scatterplot(ax = ax, x=x, y=y, palette="ch:r=-.2,d=.3_r",
                            hue=hue, size=size, sizes=(1, 200), linewidth=0, data=data)
        else:
            sns.scatterplot(ax = ax, x=x, y=y,
                            hue=hue, size=size,
                            sizes=(1, 200), linewidth=0, data=data)


    def sns_catplot(self, data, x="", y="", hue=""):

        """
        Столбчатый график seaborn

        Пример:
        sns_catplot(df, x='platform', y='total_sales', hue='year')
        """


        sns.set(style='whitegrid')

        sns.catplot(x=x, y=y, hue=hue, kind='bar', errwidth=0,
            data=data, height=5, aspect=3)



    def squared_ratio(self, df, grouper, title=''):

        """
        График соотношений

        Пример:
        squared_ratio(df, 'geography', 'Соотношение клиентов из различных стран')
        """

        df = df.groupby(grouper).size().reset_index(name='counts')
        labels = df.apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
        sizes = df['counts'].values.tolist()
        colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

        plt.figure(figsize=(10,6), dpi= 80)
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

        plt.title(title)
        plt.axis('off')
        plt.show()


    def sorted_corr(self, data, attr):

        """
        Таблица с сортировкой корреляции конкретного аттрибута

        Пример:
        sorted_corr(df, 'money'):
        """

        correlated = pd.DataFrame(data.corr()[attr].sort_values(ascending = False))
        return correlated


    def transformer(self, data, name, grouper, func):
        """

        transformer(df, 'some_stuff_to_change', 'grouper', np.mean()):

        data - датасет
        name - столбец в котором меняем значения
        grouper - столбец по которому группируем
        func - пременяемая функция mean, median и т.д.
        """
        name = name
        data.loc[data[name].isnull(), name] = data.groupby(grouper)[name].transform(func)


    def pr_curve(self, model, features_valid, target_valid):

        """
        PR-кривая

        Пример:
        pr_curve(model, features_valid, target_valid)
        """

        probabilities_valid = model.predict_proba(features_valid)
        precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])

        plt.figure(figsize=(6, 6))
        plt.step(recall, precision, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Кривая Precision-Recall')
        plt.show()

    def roc_curve(self, model, features_valid, target_valid):

        """
        ROC-кривая

        Пример:
        roc_curve(model, features_valid, target_valid)
        """

        probabilities_valid = model.predict_proba(features_valid)
        probabilities_one_valid = probabilities_valid[:, 1]

        fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid)

        plt.figure()
        plt.plot(fpr, tpr)

        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        plt.title('ROC-кривая')

        plt.show()


    def metrics_plot(self, model, features_valid, target_valid):

        """
        Выводит на экран PR-кривую и ROC-кривую

        Пример:
        metrics_plot(model, features_valid, target_valid)
        """

        probabilities_valid = model.predict_proba(features_valid)
        precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])
        fpr, tpr, thresholds = roc_curve(target_valid, probabilities_valid[:, 1])

        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        #fig, ax = plt.subplots(ncols=3)
        #fig.subplots_adjust(hspace=0.4, wspace=0.4)

        sns.lineplot(recall, precision, drawstyle='steps-post', ax=ax[0])
        ax[0].set_xlabel('Recall')
        ax[0].set_ylabel('Precision')
        ax[0].set_ylim([0.0, 1.05])
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_title('Кривая Precision-Recall')

        sns.lineplot(fpr, tpr, ax=ax[1])
        ax[1].plot([0, 1], [0, 1], linestyle='--')
        ax[1].set_xlim(0,1)
        ax[1].set_ylim(0,1)
        ax[1].set_xlabel('False Positive Rate')
        ax[1].set_ylabel('True Positive Rate')
        ax[1].set_title('ROC-кривая')


    def auc_roc(self, model, features_valid, target_valid):

        """
        Посчитывает значение ROC-AUC

        Пример:
        auc_roc(self, model, features_valid, target_valid)
        """

        probabilities_valid = model.predict_proba(features_valid)
        probabilities_one_valid = probabilities_valid[:, 1]
        auc_roc = roc_auc_score(target_valid, probabilities_one_valid)

        return auc_roc

    def upsample(self, features, target, repeat):

        """
        Дублирует объекты положительного класса и объединяет их с объектами отрицательного класса

        Пример:
        upsample(x_train, y_train, 4)
        """

        features_zeros = features[target == 0]
        features_ones = features[target == 1]
        target_zeros = target[target == 0]
        target_ones = target[target == 1]

        features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
        target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

        features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=42)

        return features_upsampled, target_upsampled


    def downsample(self, features, target, fraction):

        """
        Исключает долю объектов отрицательного класса и объединяет их с объектами положительного класса

        Пример:
        downsample(x_train, y_train, 0.5)
        """

        features_zeros = features[target == 0]
        features_ones = features[target == 1]
        target_zeros = target[target == 0]
        target_ones = target[target == 1]

        features_downsampled = pd.concat([features_zeros.sample(frac=fraction, random_state=42)] + [features_ones])
        target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=42)] + [target_ones])

        features_downsampled, target_downsampled = shuffle(features_downsampled, target_downsampled, random_state=42)

        return features_downsampled, target_downsampled


    def firstsight(self, data):

        """
        Возврашает пять первых, последних и случайных элементов датасета для дальнейшего вывода с помощью Display

        Пример:
        head, tail, sample = explorer.firstsight(df)
        """

        head = data.head(5)
        tail = data.tail(5)
        sample = data.sample(5)

        return head, tail, sample

    def clear_text(self, data, text_to_corpus):

        """
        Создает корпус, очищает текст от шумовых знаков, нормализует текст стеммингом, создает в датасете столбец с очищенным текстом и приводит его в нижний регистр

        Пример:
        df = explorer.clear_text(df, df.text)
        """
        started = time.time()
        corpus = list(text_to_corpus)
        #lemmatizer = WordNetLemmatizer()
        #lemmatized = [[lemmatizer.lemmatize(word) for word in word_tokenize(s)] for s in corpus]
        #lemm_text = [' '.join(lemma) for lemma in lemmatized]
        #clear_text = [re.sub(r'[^a-zA-Z]',' ', text) for text in lemm_text]
        pure_text = [re.sub(r'[^a-zA-Z]',' ', text) for text in corpus]
        stemmer = SnowballStemmer("english")
        stemmed = [[stemmer.stem(word) for word in word_tokenize(s)] for s in pure_text]
        stem_text = [' '.join(word) for word in stemmed]
        data['clear_text'] = stem_text
        data['clear_text'] = data['clear_text'].str.lower()
        ended = time.time()
        print('Очистка текста выполнена за {} минуты'.format((ended-started)//60))
        return data

    def tfid_features_preparation(self, features, target, train_size, language_for_stopwords):

        """
        Разделяет выборку на обучающую и тестовую, векторизирует тексты, возвращает разделенные и векторезированные признаки и тагеты

        Пример:
        x_train, x_test, y_train, y_test = explorer.tfid_features_preparation(df['clear_text'], df.toxic, 0.8, 'english', False)
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=train_size, random_state=42)

        stopwords = set(nltk_stopwords.words(language_for_stopwords))

        tfid_vect = TfidfVectorizer()

        x_train = tfid_vect.fit_transform(x_train)
        x_test = tfid_vect.transform(x_test)

        return x_train, x_test, y_train, y_test

    def reporter(self, models, score, scoring):
        started = time.time()
        report = []
        estimators = []
        predictions = []
        score_name = str(score).split(' ')[1]

        for model in models:
            print('\n', model[0], '\n')
            grid_search = self.grid_search(model[1], model[2], 5, scoring, x_train, y_train)
            print(grid_search)

            predicted = np.ravel(grid_search.predict(x_test))
            score = f1_score(y_test, predicted)
            roc_auc = self.auc_roc(grid_search, x_test, y_test)

            report.append((model[0], score, roc_auc))
            estimators.append((model[0], grid_search))
            predictions.append((model[0], predicted))
            self.metrics_plot(grid_search, model[0], x_test, y_test)
            print('\n', 'Classification report for ' + model[0], '\n\n', classification_report(y_test, predicted))

        self.final_report = pd.DataFrame(report, columns=['model', score_name, 'ROC-AUC'])
        self.best_estimator = pd.DataFrame(estimators, columns=['model', 'grid_params'])
        self.predictions = pd.DataFrame(predictions, columns=['model', 'predictions'])
        ended = time.time()
        print('Обучение с кросс-валидацей и поиском параметров выполнено за {} минуты. '.format((ended-started)//60))

    def smape(self, y_test, y_predict):

        """
        Расcчитывает метрику SMAPE

        Пример:
        smape(y_test, y_predict)
        """

        y_test, y_predict = np.array(y_test), np.array(y_predict)
        return np.mean(np.abs(y_predict - y_test) / ((np.abs(y_test) + np.abs(y_predict) + 0.1**99)/2)) * 100


    def mape(self, y_test, y_predict):

        """
        Расcчитывает метрику MAPE

        Пример:
        mape(y_test, y_predict)
        """

        y_test, y_predict = np.array(y_test), np.array(y_predict)
        return np.median((np.abs((y_test - y_predict)) / (y_test + 0.1**100)) * 100)


    def df_split(self, data, features_drop, target, test_size, random_state):

        """
        Делит датасет по заданным параметрам

        Пример:
        x_train, x_test, y_train, y_test = explorer.df_split(df_geo_1, ['id', 'product'], 'product', 0.25, 42)
        """

        feature = data.drop(features_drop, axis=1)
        target = data[target]
        x_train, x_test, y_train, y_test  = train_test_split(feature, target, test_size=test_size, random_state = random_state)
        return x_train, x_test, y_train, y_test


    def grid_search(self, model, param_grid, cv, x, y):

        """
        Поиск по сетке с заданными параметрами

        Пример:
        lr_geo_one = explorer.grid_search(lr, param_grid, 5, x_train, y_train)
        """

        grid_model = GridSearchCV(model, param_grid=param_grid, cv=cv, verbose=1, n_jobs=-1)
        grid_model.fit(x, y)
        best_estimator = grid_model.best_estimator_
        return best_estimator

    def color_styler(self, val):

        """
        Окрашивает зеленым числовые значения выше нуля, красным - ниже нуля

        Пример:

        df.style.applymap(color_styler)

        """

        color = 'green' if val > 0 else 'red'

        return 'color: %s' % color


    def derivative(f, var):
        """
        Ищем производную функции относительно заданной перменной. Аргументы в формате str.

        Пример:

        derivative('x**2', 'x')

        """

        f = sympify(f)
        var = Symbol(var)
        d = Derivative(f, var).doit()
        pprint(d)

    def ecdf(self, data):
        """Compute ECDF for a one-dimensional array of measurements."""
        # Число точек: n
        n = len(data)

        # x-data for the ECDF: x
        x = np.sort(x)

        # y-data for the ECDF: y
        y = np.arange(1, n+1) / n

        return x, y

    class Display(object):
        """
        Выводит HTML представление нескольких объектов

        Пример:
        Display(head, tail, sample)

        """
        template = """<div style="float: left; padding: 10px;">
        <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
        </div>"""
        def __init__(self, *args):
            self.args = args

        def _repr_html_(self):
            return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                             for a in self.args)

        def __repr__(self):
            return '\n\n'.join(a + '\n' + repr(eval(a))
                               for a in self.args)
