# Copyright 2018, University of Freiburg,
# Author: Anushe Glushik <anush.davtyan@jupiter.uni-freiburg.de>

import json
import codecs
import copy


class ChatbotDataset():

    def __init__(self, api_data, idx, question,
                 file_to_save, file_to_save_main):

        self.file_to_save = file_to_save
        self.file_to_save_main = file_to_save_main

        self.save_question_in_user_data(api_data, idx, question)

    def save_question_in_user_data(self, api_data, idx, question):

        """ Save a question and the data, validated
        by an Aqqu Chatbot user to a special dataset."""

        question = question.lower()
        data = self.open_users_data_file(self.file_to_save)
        data = self.add_query(data, api_data, idx, question)
        data = self.check_if_merge(data)
        self.write_renewed_data(self.file_to_save, data)

    def open_users_data_file(self, filename):

        """ Open and read the file with already
        saved quesries and answers."""

        print("Open file ", filename)
        try:
            f = open(filename, 'r')
        except IOError:
            f = open(filename, 'w+')
        try:
            data = json.load(f)
        except ValueError:
            data = {}
            data["Version"] = "Aqqu Chatbot Users Evaluated"
            data["FreebaseVersion"] = "Model Trained on 2015-08-09"
            data["Questions"] = []
        f.close()
        return data

    def add_query(self, data, api_data, idx, question):

        """ Checks if a query have been asked before, if yes - rewrite it,
        if no - add a new question-answer to a questions list."""

        add_question = True
        new_question = self.create_new_question_for_data(data,
                                                         api_data,
                                                         idx,
                                                         question)
        for q in data['Questions']:
            if q['ProcessedQuestion'] == new_question['ProcessedQuestion']:
                q = new_question
                add_question = False
        if add_question:
            data['Questions'].append(new_question)
        return data

    def create_new_question_for_data(self, data, api_data, idx, question):

        new_question = {}
        # get the id of a new question, +1 to the last id name
        try:
            new_question_id = self.get_question_id(data)
        except (IndexError, KeyError):
            new_question_id = "0"

        new_question["QuestionId"] = "AqquChatbot-" + new_question_id
        new_question["RawQuestion"] = question
        proc_question = self.get_processed_question(question)
        new_question["ProcessedQuestion"] = proc_question
        parses_list = self.get_parses_list(api_data, idx)
        new_question["Parses"] = parses_list
        return new_question

    def get_question_id(self, data):

        """ Get the id if the last question"""
        last_question = data['Questions'][-1]
        last_question_id = last_question["QuestionId"].split('-')[-1]
        new_question_id = str(int(last_question_id) + 1)
        return new_question_id

    def get_processed_question(self, question):

        """ Delete '?' from the question and add
        a whitespase before an apostroph."""

        question = question.replace('?', '')
        question = question.replace("'", " '")
        return question

    def get_parses_list(self, api_data, idx):

        """ Get the value for data['Questions']['Parses']"""

        parses_list = []
        parses_dict = {}
        parses_dict["AnnotatorId"] = None
        parses_dict["ParseId"] = None
        parses_dict["AnnotatorComment"] = {}
        parses_dict["AnnotatorComment"]["ParseQuality"] = "Complete"
        parses_dict["AnnotatorComment"]["QuestionQuality"] = "Good"
        parses_dict["AnnotatorComment"]["Confidence"] = "Normal"
        parses_dict["AnnotatorComment"]["FreeFormComment"] = \
            "First-round parse verification"
        parses_dict["Sparql"] = api_data["candidates"][idx]["sparql"]
        first_ent_match = api_data["candidates"][idx]["entity_matches"][0]
        topic_entity_mid = first_ent_match["mid"]
        parses_dict["TopicEntityMid"] = topic_entity_mid
        topic_entity_name, pot_entity_mention = self.get_topic_entity_name(
            api_data, topic_entity_mid)
        parses_dict["TopicEntityName"] = topic_entity_name
        parses_dict["PotentialTopicEntityMention"] = pot_entity_mention
        inferential_chain = []
        for ic in api_data["candidates"][idx]["relation_matches"]:
            inferential_chain.append(ic["name"])
        parses_dict["InferentialChain"] = inferential_chain
        parses_dict["Constraints"] = []
        parses_dict["Time"] = None
        parses_dict["Order"] = None
        answers = self.get_answers(api_data, idx)
        parses_dict["Answers"] = answers

        parses_list.append(parses_dict)
        return parses_list

    def get_topic_entity_name(self, api_data, topic_entity_mid):

        """ Get thetopic entity name from identified entities."""

        topic_entity_name = ""
        potential_topic_entity_mention = ""
        identified_entities = api_data["parsed_query"]["identified_entities"]
        for ie in identified_entities:
            entity_mid = ie["entity"]["mid"]
            if entity_mid == topic_entity_mid:
                topic_entity_name = ie["entity"]["name"]
                potential_topic_entity_mention = ie["raw_name"]
        return topic_entity_name, potential_topic_entity_mention

    def get_answers(self, api_data, idx):

        """ Get the answers for the question from api."""

        answers = []
        # answers found by aqqu
        answer_list = api_data["candidates"][idx]["answers"]
        for a in answer_list:
            answer_dict = {}
            answer_dict["EntityName"] = a["name"]
            # if the answer is a date - no mid for answer, only name
            # if the answer is a date - "AnswerType" is value
            # if the answer is a date - "EntityName" is null
            # and "AnswerArgument" is the "name" of api answer
            try:
                answer_dict["AnswerArgument"] = a["mid"]
            except KeyError:
                answer_dict["EntityName"] = None

            if answer_dict["EntityName"] is None:
                answer_dict["AnswerType"] = "Value"
                answer_dict["AnswerArgument"] = a["name"]
            else:
                answer_dict["AnswerType"] = "Entity"
                answer_dict["AnswerArgument"] = a["mid"]
            answers.append(answer_dict)
        return answers

    def check_if_merge(self, data):

        """ Check if merge the working file to a big data file."""

        if len(data["Questions"]) > 10:
            main_data = self.merge(data)
            self.write_renewed_data(self.file_to_save_main, main_data)
            data = {}
            data["Version"] = "Aqqu Chatbot Users Evaluated"
            data["FreebaseVersion"] = "Model Trained on 2015-08-09"
            data["Questions"] = []
        return data

    def merge(self, data):

        """ Merge working file with the big main data file.
        Check if the same question exists. If yes - replace."""

        main_data = self.open_users_data_file(self.file_to_save_main)

        if len(main_data["Questions"]) == 0:
            main_data = data
        else:
            questions = copy.deepcopy(data["Questions"])
            main_questions = main_data["Questions"]

            print("Length of questions: ", len(questions))

            for i, mq in enumerate(main_questions):
                for q in data["Questions"]:
                    print("Q")
                    if mq["ProcessedQuestion"] == q["ProcessedQuestion"]:
                        main_data["Questions"][i] = q
                        try:
                            questions.remove(q)
                        except ValueError:
                            print("Not in the list.")

            main_data["Questions"].extend(questions)
        return main_data

    def write_renewed_data(self, filename, data):

       with codecs.open(filename, 'w', encoding='utf-8') as f:
           json.dump(data, f, ensure_ascii=False, indent=4)
