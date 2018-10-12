import random
import heapq
import logging
import sys

from math import log as ln
from math import factorial as fact
from collections import deque, defaultdict
from numpy import inf

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]


def exponential(l):
    return random.expovariate(l)


def deterministic(l):
    return l


def erlang(l, k):
    return sum([exponential(l) for _ in range(k)])


def gamma(a, b):
    return random.gammavariate(a, b)


def uniform(l):
    return random.random() * l


def id_generator():
    i = 1
    while True:
        yield i
        i += 1


unique_id = id_generator()

EPS = 0.0001

DISTRIBUTIONS = {
    "M": exponential,
    "D": deterministic,
    "Ek": erlang,
    "G": gamma,
    "U": uniform
}


def mean(arr):
    if len(arr) == 0:
        return None
    return sum(arr) / len(arr)


class RandomTimeableMixin:
    def __init__(self, distribution, parameters):
        self.distribution = distribution
        self.parameters = parameters
        if distribution is None:
            self._time_left = inf
        else:
            self._time_left = distribution(*parameters)

    @property
    def time_left(self):
        return self._time_left

    def get_time(self):
        self._time_left = self.distribution(*self.parameters)

    def elapse(self, time):
        if abs(self.time_left - time) <= 0:
            self.tock()
            leftover_time = time - self.time_left
            self.get_time()
            self.elapse(leftover_time)
        else:
            self._time_left -= time

    def tock(self):
        raise NotImplementedError()


class ProducerConsumerMixin(RandomTimeableMixin):
    def __init__(self, queue, distribution, parameters, name=None):
        super(ProducerConsumerMixin, self).__init__(distribution, parameters)
        self.queue = queue
        if name is not None:
            self.name = name
        else:
            self.name = self.get_unique_name()

    def get_unique_name(self):
        raise NotImplementedError()


class Issue(RandomTimeableMixin):
    priority = 0
    fullfilled = False
    expired = False

    def __init__(self, distribution=None, parameters=None):
        super(Issue, self).__init__(distribution, parameters)
        self.max_wait_time = self.time_left
        self.waited_for = 0
        self.id = next(unique_id)

    def __repr__(self):
        return "[Issue-{}]".format(self.id)

    def tock(self):
        self.expired = True

    def elapse(self, time):
        super(Issue, self).elapse(time)
        self.waited_for += time


class Full(Exception):
    pass


class Empty(Exception):
    pass


class NoQueue(Exception):
    pass


class Producer(ProducerConsumerMixin):
    workers = []
    rejected = None
    wait_time_distribution = None
    wait_time_parameters = None

    def produce_one(self):
        self.elapsed = self.time_left
        if (self.wait_time_distribution is not None
                and self.wait_time_parameters is not None):
            return Issue(self.wait_time_distribution, self.wait_time_parameters)
        else:
            return Issue()

    def tock(self):
        issue = self.produce_one()
        available_workers = [worker for worker in self.workers if worker.available]
        if not available_workers:
            try:
                self.queue.append(issue)
            except Full:
                self.rejected.append((self.elapsed, issue))
                logger.debug(
                    "[{}] Queue is full, issue {} is rejected".format(
                        self.name,
                        issue
                    )
                )
            except NoQueue:
                self.rejected.append((self.elapsed, issue))
                logger.debug(
                    "[{}] All workers are busy, issue {} is rejected".format(
                        self.name,
                        issue
                    )
                )
        else:
            any_free_worker = random.choice(available_workers)
            logger.debug(
                "[{}] passed the issue {} to [{}]".format(
                    self.name,
                    issue,
                    any_free_worker.name
                )
            )
            any_free_worker.work_on(issue)

    def get_unique_name(self):
        return "Producer-{}".format(next(unique_id))


class Queue:
    def __init__(self, size=inf, prioritized=False, expired_queue=None):
        self.prioritized = prioritized
        self.expired_queue = expired_queue
        self.max_capacity = size
        self.size = 0
        if prioritized:
            self.q = []
        else:
            self.q = deque()

    def __iter__(self):
        return iter(self.q)

    def __len__(self):
        return self.size

    def append(self, issue):
        if self.max_capacity == 0:
            raise NoQueue()

        if self.size >= self.max_capacity:
            raise Full()

        logger.debug("Issue {} put into the queue".format(issue))
        self.size += 1
        if self.prioritized:
            heapq.heappush(self.q, (issue.priority, issue))
        else:
            self.q.append(issue)

    def popleft(self):
        if self.max_capacity == 0:
            raise NoQueue()

        if self.size == 0:
            raise Empty()

        self.size -= 1
        if self.prioritized:
            issue = heapq.heappop(self.q)
        else:
            issue = self.q.popleft()

        logger.debug("Issue {} left the queue".format(issue))
        return issue

    def remove_expired(self):
        if self.prioritized:
            tmp = self.q[:]
            self.q = []
            for issue in tmp:
                if not issue.expired:
                    heapq.heappush(self.q, (issue.priority, issue))
                else:
                    self.expired_queue.append((issue.waited_for, issue))
                    logger.debug("Issue {} expired".format(issue))
        else:
            tmp = list(self.q)
            self.q = deque()
            for issue in tmp:
                if not issue.expired:
                    self.q.append(issue)
                else:
                    self.expired_queue.append((issue.waited_for, issue))
                    logger.debug("Issue {} expired".format(issue))
        self.size = len(self.q)


class Worker(ProducerConsumerMixin):
    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__(*args, **kwargs)
        self._time_left = inf
        self.finished = None
        self.current_issue = None

    @property
    def available(self):
        return self.current_issue is None

    def work_on(self, issue):
        self.current_issue = issue
        self.get_time()
        self.elapsed = self.time_left

    def take_one(self):
        return self.queue.popleft()

    def tock(self):
        try:
            self.finished.append((self.elapsed, self.current_issue))
            self.current_issue = None
            issue = self.take_one()
            self.work_on(issue)
        except Empty:
            logger.debug("[{}] No issues in queue, resting".format(self.name))
            self.current_issue = None
            self._time_left = inf
        except NoQueue:
            logger.debug(
                "[{}] Issue {} finished, resting".format(
                    self.name,
                    self.current_issue
                )
            )
            self.current_issue = None
            self._time_left = inf

    def get_unique_name(self):
        return "Worker-{}".format(next(unique_id))


class QueueingSystem:
    FINISH_ON_ITERATION_COUNT = 'iteration_count'
    FINISH_ON_REJECTED_COUNT = 'rejected_count'
    FINISH_ON_FINISHED_COUNT = 'finished_count'
    FINISH_ON_ALL_COUNT = 'all_count'
    FINISH_ON_TIME = 'time_count'

    ERLANG = 'erlang'
    PATIENT = 'patient'
    QUEUE = 'queue'

    def __init__(self,
                 rule,
                 T=None,
                 A_params=None,
                 B_params=None,
                 T_params=None,
                 mode=FINISH_ON_TIME,
                 infinite_queue=False,
                 limit=100
                 ):
        rules = rule.split("/")
        if len(rules) < 3 or len(rules) > 4:
            raise ValueError()
        elif len(rules) == 3:
            A, B, N = rules
            K = 0 if not infinite_queue else inf
        else:
            A, B, N, K = rules
            K = int(K)

        self.income_intensity = A_params[0]
        self.work_intensity = B_params[0]

        producers_distribution = DISTRIBUTIONS[A]
        workers_distribution = DISTRIBUTIONS[B]
        if T:
            issue_wait_time_distribution = DISTRIBUTIONS[T]
            self.impatient_mode = True

        self.mode = mode
        self.limit = limit
        self.elapsed = 1e-7
        self.iteration = 0

        self.stats = {
            'in_queue': [],
            'in_system': []
        }

        self.finished_queue = deque()
        self.rejected_queue = deque()
        self.expired_queue = deque()

        self.queue = Queue(size=K, expired_queue=self.expired_queue)

        self.workers = []
        self.producers = []

        for _ in range(int(N)):
            w = Worker(self.queue, workers_distribution, B_params)
            w.finished = self.finished_queue
            self.workers.append(w)

        self.workers_busy_distribution = defaultdict(float)
        self.queue_busy_distribution = defaultdict(float)

        p = Producer(self.queue, producers_distribution, A_params)
        if T:
            p.wait_time_distribution = issue_wait_time_distribution
            p.wait_time_parameters = T_params
        p.workers = self.workers
        p.rejected = self.rejected_queue
        self.producer = p

    @property
    def p(self):
        p = self.income_intensity / self.work_intensity
        if self.type == self.PATIENT and p >= 1:
            raise ValueError("Infinite queue has no limit probability")
        return p

    @property
    def rejected_count(self):
        return len(self.rejected_queue)

    @property
    def finished_count(self):
        return len(self.finished_queue)

    @property
    def all_count(self):
        return self.rejected_count + self.finished_count + self.processed_count

    @property
    def processed_count(self):
        return sum([1 for worker in self.workers if not worker.available])

    @property
    def should_stop(self):
        if self.mode == self.FINISH_ON_ITERATION_COUNT:
            return self.iteration >= self.limit
        elif self.mode == self.FINISH_ON_REJECTED_COUNT:
            return self.rejected_count >= self.limit
        elif self.mode == self.FINISH_ON_FINISHED_COUNT:
            return self.finished_count >= self.limit
        elif self.mode == self.FINISH_ON_ALL_COUNT:
            return self.all_count >= self.limit
        elif self.mode == self.FINISH_ON_TIME:
            return self.elapsed >= self.limit

    @property
    def type(self):
        q = self.queue.max_capacity
        if q == 0:
            return self.ERLANG
        elif q == inf:
            return self.PATIENT
        else:
            return self.QUEUE

    @property
    def verbose_type(self):
        worker_count = len(self.workers)
        queue_size = self.queue.max_capacity
        impatient = self.impatient_mode
        result = "{}канальная СМО".format(str(worker_count) + "-" if worker_count != 1 else "Одно")

        if queue_size == 0:
            result += " с отказами"
        elif queue_size == inf:
            result += " без отказов (с бесконечной очередью)"
        else:
            result += " с конечной очередью"

        if impatient:
            result += " и нетерпеливыми клиентами"

        return result

    @property
    def free_probability(self):
        count = len(self.workers)
        worker_denominator = sum([self.p ** n / fact(n) for n in range(count + 1)])
        if self.queue.max_capacity == inf:
            queue_denominator = 1 / (1 - self.p)
        else:
            queue_denominator = sum([self.p ** n for n in range(self.queue.max_capacity)])
        return 1 / (worker_denominator + self.p ** (count + 1) * queue_denominator / fact(count))

    @property
    def reject_probability(self):
        n = len(self.workers)
        return (self.p ** n) / fact(n) * self.free_probability

    @property
    def absolute_bandwith(self):
        return self.relative_bandwith * self.income_intensity

    @property
    def relative_bandwith(self):
        return 1 - self.reject_probability

    @property
    def busy_mean(self):
        return self.absolute_bandwith / self.work_intensity

    @property
    def average_issue_system(self):
        return self.p / (1 - self.p)

    @property
    def average_time_system(self):
        return self.average_issue_system / self.income_intensity

    @property
    def average_issue_queue(self):
        return self.p ** 2 / (1 - self.p)

    @property
    def average_time_queue(self):
        return self.average_issue_queue / self.income_intensity

    def start(self):
        while not self.should_stop:
            yield self.tick()
        raise StopIteration()

    def tick(self):
        min_step = min(
            [self.producer.time_left]
            + [w.time_left for w in self.workers]
            + [i.time_left for i in self.queue]
        )

        self.step = min_step
        state = self.get_state()

        logger.debug("[{}] has {} secons left".format(self.producer.name, self.producer.time_left))
        for p in self.workers:
            logger.debug("[{}] has {} secons left".format(p.name, p.time_left))

        for worker in self.workers:
            worker.elapse(min_step)

        for issue in self.queue:
            issue.elapse(min_step)

        self.queue.remove_expired()

        self.producer.elapse(min_step)

        self.elapsed += min_step
        self.iteration += 1

        self.update_stats(min_step)

        return {"stats": self.get_stats(), "state": state}

    def get_state(self):
        return {
            "workers": {w.name: str(w.current_issue) for w in self.workers},
            "workers_busy": {w.name: min(w.time_left, 90000) for w in self.workers},
            "queue": [str(i) for i in self.queue],
            "next_in": self.producer.time_left,
            "step": self.step,
            "finished": [str(i[1]) for i in self.finished_queue],
            "rejected": [str(i[1]) for i in self.rejected_queue],
            "expired": [str(i[1]) for i in self.expired_queue],
            "time_dist": dict(self.workers_busy_distribution),
            "queue_dist": dict(self.queue_busy_distribution),
        }

    def update_stats(self, step):
        self.workers_busy_distribution[self.processed_count] += step
        self.queue_busy_distribution[self.queue.size] += step
        for prop in self.get_props():
            self.stats[prop] = getattr(self, 'get_actual_' + prop)(step)

    def get_actual_absolute_bandwith(self, step):
        return self.finished_count / self.elapsed

    def get_actual_relative_bandwith(self, step):
        return self.finished_count / self.all_count

    def get_actual_busy_mean(self, step):
        return sum([k * (v / self.elapsed) for k, v in self.workers_busy_distribution.items()])

    def get_actual_reject_probability(self, step):
        return self.rejected_count / self.all_count

    def get_actual_free_probability(self, step):
        return self.workers_busy_distribution[0] / self.elapsed

    def get_actual_average_issue_system(self, step):
        return mean(self.stats['in_system'])

    def get_actual_average_time_system(self, step):
        return mean([k + v.waited_for for k, v in self.finished_queue])

    def get_actual_average_issue_queue(self, step):
        return mean(self.stats['in_queue'])

    def get_actual_average_time_queue(self, step):
        return mean([v.waited_for for k, v in self.finished_queue])

    def get_props(self):
        common = ['absolute_bandwith', 'relative_bandwith', 'reject_probability',
                  'busy_mean', 'free_probability']
        queue = ['average_issue_system', 'average_time_system',
                 'average_issue_queue', 'average_time_queue', 'free_probability']
        return queue

    def get_actual_stats(self):
        return {prop: self.stats[prop] for prop in self.get_props()}

    def get_stats_for_props(self, props):
        return {prop: getattr(self, prop) for prop in props}

    def get_stats(self):
        common = {
            "iteration": self.iteration,
            "elapsed": self.elapsed,
            "total_issues": self.all_count,
            "workers_time_distribution": {k: v / self.elapsed for k, v in self.workers_busy_distribution.items()}
        }
        try:
            expected = self.get_stats_for_props(self.get_props())
        except ValueError:
            expected = None
        actual = self.get_actual_stats()
        common['expected'] = expected
        common['actual'] = actual

        return common
